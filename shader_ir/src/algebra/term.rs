//! Scalar terms: the right-hand side of an index-notation definition.

use super::affine::{Affine, IndexVar, Sym};
use std::collections::{BTreeSet, HashMap};
use std::ops;

/// A tensor value of a [`Region`](super::Region).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub(crate) u32);

/// A runtime f32 parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScalarParam(pub(crate) u32);

impl ScalarParam {
    /// Position among the region's scalar parameters, in creation order.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Recip,
    Sin,
    Cos,
    /// The argument's value, rounded to f32 on its own: a product here does not
    /// contract with the operation that reads it, as when it was stored between them.
    Round,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
}

/// The association in which a reduction combines its terms.
///
/// Part of the term, because floating-point combination is not associative: two
/// reductions with the same terms in different orders are different terms. A schedule
/// must realize the order a reduction names; changing it is
/// [`Exactness::ReductionOrder`](super::Exactness::ReductionOrder).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ReduceOrder {
    /// One accumulator, from the identity, in ascending index order.
    #[default]
    Sequential,
    /// `lanes` partial reductions, then a pairwise tree over them.
    ///
    /// Lane `l` owns indices `l, l + lanes, l + 2·lanes, …`, dealt round-robin over
    /// `accumulators` accumulators that each start from the identity; its partial is
    /// the left fold of its accumulators. The tree then combines partial `l` with
    /// partial `l + s` for `s = lanes/2, lanes/4, …, 1`. `lanes` is a power of two.
    Lanes { lanes: u32, accumulators: u32 },
}

impl ReduceOp {
    pub fn identity(self) -> f32 {
        match self {
            ReduceOp::Sum => 0.0,
            ReduceOp::Max => f32::NEG_INFINITY,
            ReduceOp::Min => f32::INFINITY,
        }
    }

    pub fn combine(self, acc: f32, term: f32) -> f32 {
        match self {
            ReduceOp::Sum => acc + term,
            ReduceOp::Max => acc.max(term),
            ReduceOp::Min => acc.min(term),
        }
    }
}

/// Comparison of two affine expressions in a [`Term::Select`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Lt,
    Le,
    Eq,
    Ne,
}

impl CmpOp {
    pub fn holds(self, lhs: i64, rhs: i64) -> bool {
        match self {
            CmpOp::Lt => lhs < rhs,
            CmpOp::Le => lhs <= rhs,
            CmpOp::Eq => lhs == rhs,
            CmpOp::Ne => lhs != rhs,
        }
    }
}

/// An f32 scalar expression over index variables.
///
/// The tree is the evaluation order: floating-point arithmetic is not associative, so
/// `a + (b + c)` and `(a + b) + c` are different terms.
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    Lit(f32),
    Scalar(ScalarParam),
    /// An affine index converted to f32.
    IndexValue(Affine),
    Read {
        value: ValueId,
        index: Vec<Affine>,
    },
    Unary {
        op: UnaryOp,
        arg: Box<Term>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Term>,
        rhs: Box<Term>,
    },
    /// `lhs cmp rhs ? then : otherwise`. Only the taken branch is evaluated.
    Select {
        lhs: Affine,
        cmp: CmpOp,
        rhs: Affine,
        then: Box<Term>,
        otherwise: Box<Term>,
    },
    /// `op` over `body` for `index` in `0..extent`, combined in `order`.
    Reduce {
        op: ReduceOp,
        index: IndexVar,
        extent: u32,
        order: ReduceOrder,
        body: Box<Term>,
    },
}

/// A subterm that does not depend on some index bound around it.
#[derive(Debug, Clone, PartialEq)]
pub struct Invariant<'a> {
    pub term: &'a Term,
    /// The indices it depends on. Every other enclosing index can vary without
    /// changing its value, so it can be computed once outside their loops.
    pub free: BTreeSet<IndexVar>,
}

impl Term {
    pub fn lit(value: f32) -> Term {
        Term::Lit(value)
    }

    pub fn scalar(param: ScalarParam) -> Term {
        Term::Scalar(param)
    }

    /// Argument `n` of an operation's element-wise term, such as a
    /// [`Map`](super::Map) body.
    pub fn arg(n: usize) -> Term {
        Term::Read {
            value: ValueId(n as u32),
            index: Vec::new(),
        }
    }

    pub fn index_value(index: impl Into<Affine>) -> Term {
        Term::IndexValue(index.into())
    }

    pub fn read<A: Into<Affine>>(value: ValueId, index: impl IntoIterator<Item = A>) -> Term {
        Term::Read {
            value,
            index: index.into_iter().map(Into::into).collect(),
        }
    }

    pub fn unary(op: UnaryOp, arg: Term) -> Term {
        Term::Unary { op, arg: Box::new(arg) }
    }

    pub fn binary(op: BinaryOp, lhs: Term, rhs: Term) -> Term {
        Term::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn select(lhs: impl Into<Affine>, cmp: CmpOp, rhs: impl Into<Affine>, then: Term, otherwise: Term) -> Term {
        Term::Select {
            lhs: lhs.into(),
            cmp,
            rhs: rhs.into(),
            then: Box::new(then),
            otherwise: Box::new(otherwise),
        }
    }

    /// A [`ReduceOrder::Sequential`] reduction.
    pub fn reduce(op: ReduceOp, index: IndexVar, extent: u32, body: Term) -> Term {
        Term::reduce_in(op, ReduceOrder::Sequential, index, extent, body)
    }

    pub fn reduce_in(op: ReduceOp, order: ReduceOrder, index: IndexVar, extent: u32, body: Term) -> Term {
        Term::Reduce {
            op,
            index,
            extent,
            order,
            body: Box::new(body),
        }
    }

    pub fn sum(index: IndexVar, extent: u32, body: Term) -> Term {
        Term::reduce(ReduceOp::Sum, index, extent, body)
    }

    pub fn min(self, rhs: Term) -> Term {
        Term::binary(BinaryOp::Min, self, rhs)
    }

    pub fn max(self, rhs: Term) -> Term {
        Term::binary(BinaryOp::Max, self, rhs)
    }

    pub fn abs(self) -> Term {
        Term::unary(UnaryOp::Abs, self)
    }

    pub fn exp(self) -> Term {
        Term::unary(UnaryOp::Exp, self)
    }

    pub fn log(self) -> Term {
        Term::unary(UnaryOp::Log, self)
    }

    pub fn sqrt(self) -> Term {
        Term::unary(UnaryOp::Sqrt, self)
    }

    pub fn recip(self) -> Term {
        Term::unary(UnaryOp::Recip, self)
    }

    pub fn sin(self) -> Term {
        Term::unary(UnaryOp::Sin, self)
    }

    pub fn cos(self) -> Term {
        Term::unary(UnaryOp::Cos, self)
    }

    /// Index variables mentioned outside any reduction that binds them.
    pub fn free_indices(&self) -> BTreeSet<IndexVar> {
        let mut free = BTreeSet::new();
        self.collect_free(&mut Vec::new(), &mut free);
        free
    }

    fn collect_free(&self, bound: &mut Vec<IndexVar>, free: &mut BTreeSet<IndexVar>) {
        let note = |a: &Affine, bound: &[IndexVar], free: &mut BTreeSet<IndexVar>| {
            free.extend(a.indices().filter(|v| !bound.contains(v)));
        };
        match self {
            Term::Lit(_) | Term::Scalar(_) => {}
            Term::IndexValue(a) => note(a, bound, free),
            Term::Read { index, .. } => index.iter().for_each(|a| note(a, bound, free)),
            Term::Unary { arg, .. } => arg.collect_free(bound, free),
            Term::Binary { lhs, rhs, .. } => {
                lhs.collect_free(bound, free);
                rhs.collect_free(bound, free);
            }
            Term::Select {
                lhs,
                rhs,
                then,
                otherwise,
                ..
            } => {
                note(lhs, bound, free);
                note(rhs, bound, free);
                then.collect_free(bound, free);
                otherwise.collect_free(bound, free);
            }
            Term::Reduce { index, body, .. } => {
                bound.push(*index);
                body.collect_free(bound, free);
                bound.pop();
            }
        }
    }

    /// Distinct reduction binders, in order of first appearance.
    pub fn binders(&self) -> Vec<IndexVar> {
        let mut out = Vec::new();
        self.visit(&mut |t| {
            if let Term::Reduce { index, .. } = t {
                if !out.contains(index) {
                    out.push(*index);
                }
            }
        });
        out
    }

    /// Reductions not nested in another reduction, in evaluation order.
    pub fn reductions(&self) -> Vec<&Term> {
        let mut out = Vec::new();
        self.collect_reductions(&mut out);
        out
    }

    fn collect_reductions<'a>(&'a self, out: &mut Vec<&'a Term>) {
        match self {
            Term::Reduce { .. } => out.push(self),
            _ => self.for_each_child(&mut |c| c.collect_reductions(out)),
        }
    }

    /// Calls `f` with the order of every reduction, nested ones too.
    pub(crate) fn for_each_reduction(&self, f: &mut dyn FnMut(ReduceOrder)) {
        self.visit(&mut |t| {
            if let Term::Reduce { order, .. } = t {
                f(*order);
            }
        });
    }

    /// Calls `f` for every read, in evaluation order.
    pub fn for_each_read(&self, f: &mut dyn FnMut(ValueId, &[Affine])) {
        self.visit(&mut |t| {
            if let Term::Read { value, index } = t {
                f(*value, index);
            }
        });
    }

    /// Structural equality up to consistent renaming of reduction binders.
    pub fn alpha_eq(&self, other: &Term) -> bool {
        alpha_eq(self, other, &mut Vec::new())
    }

    /// Maximal subterms that do not depend on every index bound around them.
    ///
    /// `domain` holds the indices bound outside the term, usually the domain of the
    /// definition it belongs to. Loads, literals and parameters are not reported: they
    /// are cheaper to repeat than to hoist. A reported term's own subterms may be
    /// invariant with respect to more indices; call again on it with its `free` set.
    pub fn invariants(&self, domain: &[IndexVar]) -> Vec<Invariant<'_>> {
        let mut out = Vec::new();
        let mut bound = domain.to_vec();
        self.collect_invariants(&mut bound, &mut out);
        out
    }

    fn collect_invariants<'a>(&'a self, bound: &mut Vec<IndexVar>, out: &mut Vec<Invariant<'a>>) {
        let trivial = matches!(
            self,
            Term::Lit(_) | Term::Scalar(_) | Term::IndexValue(_) | Term::Read { .. }
        );
        let free = self.free_indices();
        if !trivial && bound.iter().any(|v| !free.contains(v)) {
            out.push(Invariant { term: self, free });
            return;
        }
        match self {
            Term::Reduce { index, body, .. } => {
                bound.push(*index);
                body.collect_invariants(bound, out);
                bound.pop();
            }
            _ => self.for_each_child(&mut |c| c.collect_invariants(bound, out)),
        }
    }

    pub(crate) fn for_each_child<'a>(&'a self, f: &mut dyn FnMut(&'a Term)) {
        match self {
            Term::Lit(_) | Term::Scalar(_) | Term::IndexValue(_) | Term::Read { .. } => {}
            Term::Unary { arg, .. } => f(arg),
            Term::Binary { lhs, rhs, .. } => {
                f(lhs);
                f(rhs);
            }
            Term::Select { then, otherwise, .. } => {
                f(then);
                f(otherwise);
            }
            Term::Reduce { body, .. } => f(body),
        }
    }

    fn visit<'a>(&'a self, f: &mut dyn FnMut(&'a Term)) {
        f(self);
        self.for_each_child(&mut |c| c.visit(f));
    }

    /// Rebuilds every child with `f`, keeping this node.
    pub(crate) fn map_children(self, f: &mut dyn FnMut(Term) -> Term) -> Term {
        match self {
            leaf @ (Term::Lit(_) | Term::Scalar(_) | Term::IndexValue(_) | Term::Read { .. }) => leaf,
            Term::Unary { op, arg } => Term::Unary {
                op,
                arg: Box::new(f(*arg)),
            },
            Term::Binary { op, lhs, rhs } => Term::Binary {
                op,
                lhs: Box::new(f(*lhs)),
                rhs: Box::new(f(*rhs)),
            },
            Term::Select {
                lhs,
                cmp,
                rhs,
                then,
                otherwise,
            } => Term::Select {
                lhs,
                cmp,
                rhs,
                then: Box::new(f(*then)),
                otherwise: Box::new(f(*otherwise)),
            },
            Term::Reduce {
                op,
                index,
                extent,
                order,
                body,
            } => Term::Reduce {
                op,
                index,
                extent,
                order,
                body: Box::new(f(*body)),
            },
        }
    }

    /// Replaces every read, innermost first, with `f(value, index)`.
    pub(crate) fn map_reads(self, f: &mut dyn FnMut(ValueId, Vec<Affine>) -> Term) -> Term {
        match self.map_children(&mut |c| c.map_reads(&mut *f)) {
            Term::Read { value, index } => f(value, index),
            other => other,
        }
    }

    /// Renames every symbol, scalar and read value; for moving a term between regions.
    pub(crate) fn rename(
        &self,
        sym: &dyn Fn(Sym) -> Sym,
        scalar: &dyn Fn(ScalarParam) -> ScalarParam,
        value: &dyn Fn(ValueId) -> ValueId,
    ) -> Term {
        let go = |t: &Term| t.rename(sym, scalar, value);
        match self {
            Term::Lit(_) => self.clone(),
            Term::Scalar(s) => Term::Scalar(scalar(*s)),
            Term::IndexValue(a) => Term::IndexValue(a.rename(sym)),
            Term::Read { value: v, index } => Term::Read {
                value: value(*v),
                index: index.iter().map(|a| a.rename(sym)).collect(),
            },
            Term::Unary { op, arg } => Term::unary(*op, go(arg)),
            Term::Binary { op, lhs, rhs } => Term::binary(*op, go(lhs), go(rhs)),
            Term::Select {
                lhs,
                cmp,
                rhs,
                then,
                otherwise,
            } => Term::select(lhs.rename(sym), *cmp, rhs.rename(sym), go(then), go(otherwise)),
            Term::Reduce {
                op,
                index,
                extent,
                order,
                body,
            } => {
                let Sym::Index(index) = sym(Sym::Index(*index)) else {
                    unreachable!("an index variable renames to an index variable")
                };
                Term::reduce_in(*op, *order, index, *extent, go(body))
            }
        }
    }

    /// Substitutes free indices through `free`, and renames reduction binders through
    /// `binders`. The caller keeps the two disjoint so no binder captures a free index.
    pub(crate) fn instantiate(
        &self,
        free: &dyn Fn(IndexVar) -> Option<Affine>,
        binders: &HashMap<IndexVar, IndexVar>,
    ) -> Term {
        let map = |v: IndexVar| binders.get(&v).map(|&b| Affine::from(b)).or_else(|| free(v));
        let affine = |a: &Affine| a.substitute(&map);
        match self {
            Term::Lit(_) | Term::Scalar(_) => self.clone(),
            Term::IndexValue(a) => Term::IndexValue(affine(a)),
            Term::Read { value, index } => Term::Read {
                value: *value,
                index: index.iter().map(affine).collect(),
            },
            Term::Unary { op, arg } => Term::unary(*op, arg.instantiate(free, binders)),
            Term::Binary { op, lhs, rhs } => {
                Term::binary(*op, lhs.instantiate(free, binders), rhs.instantiate(free, binders))
            }
            Term::Select {
                lhs,
                cmp,
                rhs,
                then,
                otherwise,
            } => Term::select(
                affine(lhs),
                *cmp,
                affine(rhs),
                then.instantiate(free, binders),
                otherwise.instantiate(free, binders),
            ),
            Term::Reduce {
                op,
                index,
                extent,
                order,
                body,
            } => Term::reduce_in(
                *op,
                *order,
                binders.get(index).copied().unwrap_or(*index),
                *extent,
                body.instantiate(free, binders),
            ),
        }
    }
}

/// `pairs` holds the binders in scope, innermost last, as `(a's binder, b's binder)`.
fn alpha_eq(a: &Term, b: &Term, pairs: &mut Vec<(IndexVar, IndexVar)>) -> bool {
    match (a, b) {
        (Term::Lit(x), Term::Lit(y)) => x.to_bits() == y.to_bits(),
        (Term::Scalar(x), Term::Scalar(y)) => x == y,
        (Term::IndexValue(x), Term::IndexValue(y)) => affine_alpha_eq(x, y, pairs),
        (Term::Read { value: v, index: x }, Term::Read { value: w, index: y }) => {
            v == w && x.len() == y.len() && x.iter().zip(y).all(|(x, y)| affine_alpha_eq(x, y, pairs))
        }
        (Term::Unary { op: p, arg: x }, Term::Unary { op: q, arg: y }) => p == q && alpha_eq(x, y, pairs),
        (
            Term::Binary {
                op: p,
                lhs: x0,
                rhs: x1,
            },
            Term::Binary {
                op: q,
                lhs: y0,
                rhs: y1,
            },
        ) => p == q && alpha_eq(x0, y0, pairs) && alpha_eq(x1, y1, pairs),
        (
            Term::Select {
                lhs: xl,
                cmp: p,
                rhs: xr,
                then: xt,
                otherwise: xo,
            },
            Term::Select {
                lhs: yl,
                cmp: q,
                rhs: yr,
                then: yt,
                otherwise: yo,
            },
        ) => {
            p == q
                && affine_alpha_eq(xl, yl, pairs)
                && affine_alpha_eq(xr, yr, pairs)
                && alpha_eq(xt, yt, pairs)
                && alpha_eq(xo, yo, pairs)
        }
        (
            Term::Reduce {
                op: p,
                index: x,
                extent: m,
                order: s,
                body: xb,
            },
            Term::Reduce {
                op: q,
                index: y,
                extent: n,
                order: t,
                body: yb,
            },
        ) => {
            if p != q || m != n || s != t {
                return false;
            }
            pairs.push((*x, *y));
            let equal = alpha_eq(xb, yb, pairs);
            pairs.pop();
            equal
        }
        _ => false,
    }
}

fn affine_alpha_eq(a: &Affine, b: &Affine, pairs: &[(IndexVar, IndexVar)]) -> bool {
    if a.constant_term() != b.constant_term() || a.terms().len() != b.terms().len() {
        return false;
    }
    let mut mapped = Vec::with_capacity(a.terms().len());
    for &(sym, coefficient) in a.terms() {
        let sym = match sym {
            Sym::Index(v) => match pairs.iter().rev().find(|(x, _)| *x == v) {
                Some(&(_, w)) => Sym::Index(w),
                // Free in `a` but bound in `b`: the names coincide, the variables do not.
                None if pairs.iter().any(|&(_, w)| w == v) => return false,
                None => sym,
            },
            Sym::Param(_) => sym,
        };
        mapped.push((sym, coefficient));
    }
    mapped.sort_unstable_by_key(|&(sym, _)| sym);
    mapped == b.terms()
}

impl ops::Add for Term {
    type Output = Term;

    fn add(self, rhs: Term) -> Term {
        Term::binary(BinaryOp::Add, self, rhs)
    }
}

impl ops::Sub for Term {
    type Output = Term;

    fn sub(self, rhs: Term) -> Term {
        Term::binary(BinaryOp::Sub, self, rhs)
    }
}

impl ops::Mul for Term {
    type Output = Term;

    fn mul(self, rhs: Term) -> Term {
        Term::binary(BinaryOp::Mul, self, rhs)
    }
}

impl ops::Div for Term {
    type Output = Term;

    fn div(self, rhs: Term) -> Term {
        Term::binary(BinaryOp::Div, self, rhs)
    }
}

impl ops::Neg for Term {
    type Output = Term;

    fn neg(self) -> Term {
        Term::unary(UnaryOp::Neg, self)
    }
}
