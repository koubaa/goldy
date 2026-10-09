//! Integer affine expressions over index variables and index parameters.

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// An index variable, bound by a defined value's domain or by a reduction.
///
/// A variable may be reused across definitions, and by sibling reductions, but no
/// reduction rebinds a variable already bound where it appears.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IndexVar(pub(crate) u32);

/// A runtime integer with a declared range, such as a cache position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IndexParam(pub(crate) u32);

impl IndexParam {
    /// Position among the region's index parameters, in creation order.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A symbol an [`Affine`] expression may mention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Sym {
    Index(IndexVar),
    Param(IndexParam),
}

/// `constant + Σ coefficient·symbol`.
///
/// Terms are sorted by symbol and no coefficient is zero, so two expressions are equal
/// exactly when they are the same function.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Affine {
    constant: i64,
    terms: Vec<(Sym, i64)>,
}

impl Affine {
    pub fn constant(value: i64) -> Self {
        Self {
            constant: value,
            terms: Vec::new(),
        }
    }

    pub fn sym(sym: Sym) -> Self {
        Self {
            constant: 0,
            terms: vec![(sym, 1)],
        }
    }

    pub fn constant_term(&self) -> i64 {
        self.constant
    }

    pub fn terms(&self) -> &[(Sym, i64)] {
        &self.terms
    }

    pub fn as_constant(&self) -> Option<i64> {
        self.terms.is_empty().then_some(self.constant)
    }

    pub fn coefficient(&self, sym: Sym) -> i64 {
        self.terms
            .binary_search_by_key(&sym, |&(s, _)| s)
            .map_or(0, |at| self.terms[at].1)
    }

    pub fn mentions(&self, sym: Sym) -> bool {
        self.coefficient(sym) != 0
    }

    /// Index variables with a nonzero coefficient, in ascending order.
    pub fn indices(&self) -> impl Iterator<Item = IndexVar> + '_ {
        self.terms.iter().filter_map(|&(sym, _)| match sym {
            Sym::Index(v) => Some(v),
            Sym::Param(_) => None,
        })
    }

    /// Replaces each index variable for which `map` answers with that expression.
    pub fn substitute(&self, map: &dyn Fn(IndexVar) -> Option<Affine>) -> Affine {
        let mut out = Affine::constant(self.constant);
        for &(sym, coefficient) in &self.terms {
            let replacement = match sym {
                Sym::Index(v) => map(v),
                Sym::Param(_) => None,
            };
            out = out + replacement.unwrap_or_else(|| Affine::sym(sym)) * coefficient;
        }
        out
    }

    /// Replaces every symbol with `rename(symbol)`.
    pub fn rename(&self, rename: &dyn Fn(Sym) -> Sym) -> Affine {
        let mut out = Affine::constant(self.constant);
        for &(sym, coefficient) in &self.terms {
            out.insert(rename(sym), coefficient);
        }
        out
    }

    pub fn eval(&self, value: &dyn Fn(Sym) -> i64) -> i64 {
        self.terms
            .iter()
            .fold(self.constant, |acc, &(sym, coefficient)| acc + coefficient * value(sym))
    }

    /// Inclusive bounds given inclusive bounds for every symbol, or `None` when a
    /// symbol has none.
    pub fn bounds(&self, range: &dyn Fn(Sym) -> Option<(i64, i64)>) -> Option<(i64, i64)> {
        let mut lo = self.constant;
        let mut hi = self.constant;
        for &(sym, coefficient) in &self.terms {
            let (a, b) = range(sym)?;
            let (x, y) = (coefficient.saturating_mul(a), coefficient.saturating_mul(b));
            lo = lo.saturating_add(x.min(y));
            hi = hi.saturating_add(x.max(y));
        }
        Some((lo, hi))
    }

    pub(crate) fn write<'a>(&self, f: &mut fmt::Formatter<'_>, name: &dyn Fn(Sym) -> &'a str) -> fmt::Result {
        for (at, &(sym, coefficient)) in self.terms.iter().enumerate() {
            match (at == 0, coefficient < 0) {
                (true, true) => f.write_str("-")?,
                (true, false) => {}
                (false, true) => f.write_str(" - ")?,
                (false, false) => f.write_str(" + ")?,
            }
            let magnitude = coefficient.unsigned_abs();
            if magnitude != 1 {
                write!(f, "{magnitude}*")?;
            }
            f.write_str(name(sym))?;
        }
        match (self.terms.is_empty(), self.constant) {
            (true, c) => write!(f, "{c}"),
            (false, 0) => Ok(()),
            (false, c) if c > 0 => write!(f, " + {c}"),
            (false, c) => write!(f, " - {}", c.unsigned_abs()),
        }
    }

    fn insert(&mut self, sym: Sym, coefficient: i64) {
        match self.terms.binary_search_by_key(&sym, |&(s, _)| s) {
            Ok(at) => {
                self.terms[at].1 += coefficient;
                if self.terms[at].1 == 0 {
                    self.terms.remove(at);
                }
            }
            Err(at) if coefficient != 0 => self.terms.insert(at, (sym, coefficient)),
            Err(_) => {}
        }
    }
}

impl Add for Affine {
    type Output = Affine;

    fn add(mut self, rhs: Affine) -> Affine {
        self.constant += rhs.constant;
        for (sym, coefficient) in rhs.terms {
            self.insert(sym, coefficient);
        }
        self
    }
}

impl Sub for Affine {
    type Output = Affine;

    fn sub(self, rhs: Affine) -> Affine {
        self + -rhs
    }
}

impl Neg for Affine {
    type Output = Affine;

    fn neg(self) -> Affine {
        self * -1
    }
}

impl Mul<i64> for Affine {
    type Output = Affine;

    fn mul(mut self, k: i64) -> Affine {
        if k == 0 {
            return Affine::constant(0);
        }
        self.constant *= k;
        for (_, coefficient) in &mut self.terms {
            *coefficient *= k;
        }
        self
    }
}

impl Add<i64> for Affine {
    type Output = Affine;

    fn add(mut self, c: i64) -> Affine {
        self.constant += c;
        self
    }
}

impl Sub<i64> for Affine {
    type Output = Affine;

    fn sub(mut self, c: i64) -> Affine {
        self.constant -= c;
        self
    }
}

impl From<i64> for Affine {
    fn from(value: i64) -> Self {
        Affine::constant(value)
    }
}

impl From<Sym> for Affine {
    fn from(sym: Sym) -> Self {
        Affine::sym(sym)
    }
}

impl From<IndexVar> for Affine {
    fn from(v: IndexVar) -> Self {
        Affine::sym(Sym::Index(v))
    }
}

impl From<IndexParam> for Affine {
    fn from(p: IndexParam) -> Self {
        Affine::sym(Sym::Param(p))
    }
}

macro_rules! symbol_arithmetic {
    ($ty:ty) => {
        impl Add<i64> for $ty {
            type Output = Affine;

            fn add(self, c: i64) -> Affine {
                Affine::from(self) + c
            }
        }

        impl Sub<i64> for $ty {
            type Output = Affine;

            fn sub(self, c: i64) -> Affine {
                Affine::from(self) - c
            }
        }

        impl Mul<i64> for $ty {
            type Output = Affine;

            fn mul(self, k: i64) -> Affine {
                Affine::from(self) * k
            }
        }
    };
}

symbol_arithmetic!(IndexVar);
symbol_arithmetic!(IndexParam);

#[cfg(test)]
mod tests {
    use super::*;

    const I: IndexVar = IndexVar(0);
    const K: IndexVar = IndexVar(1);
    const POS: IndexParam = IndexParam(0);

    #[test]
    fn canonical_form_cancels_and_sorts() {
        let a = (K * 3 + 2) + (I - 1) - Affine::from(K) * 3;
        assert_eq!(a, I + 1);
        assert_eq!((I - 5) - Affine::from(I), Affine::constant(-5));
        assert_eq!((Affine::from(K) + Affine::from(I)).terms()[0].0, Sym::Index(I));
    }

    #[test]
    fn substitution_composes_maps() {
        let row = POS * 48;
        let access = Affine::from(I) + row.clone();
        let composed = access.substitute(&|v| (v == I).then(|| K - 2));
        assert_eq!(composed, Affine::from(K) + row - 2);
    }

    #[test]
    fn bounds_follow_coefficient_sign() {
        let a = Affine::from(I) * -2 + 10;
        let range = |sym: Sym| match sym {
            Sym::Index(_) => Some((0, 3)),
            Sym::Param(_) => None,
        };
        assert_eq!(a.bounds(&range), Some((4, 10)));
        assert_eq!((POS + 1).bounds(&range), None);
    }
}
