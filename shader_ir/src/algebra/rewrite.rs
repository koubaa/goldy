//! Rewrites of a region, each a law with a stated [`Exactness`].

use super::affine::{Affine, IndexVar, Sym};
use super::region::{Definition, Region, Role};
use super::term::{BinaryOp, Term, UnaryOp, ValueId};
use std::collections::HashMap;
use std::fmt;

/// What a rewrite preserves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Exactness {
    /// Every scalar operation on every element, and each reduction's terms in order.
    /// [`Region::evaluate`] gives bit-identical results before and after.
    Exact,
    /// Each reduction's terms, in any association.
    ReductionOrder,
    /// Only the real-number value. Needs an explicit relaxed numerical policy.
    RealAlgebra,
}

/// A local law applied throughout one definition by [`Region::apply`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Law {
    /// `f(c ? a : b) = c ? f(a) : f(b)` for an element-wise `f`, and
    /// `reduce_k(c ? a : b) = c ? reduce_k(a) : reduce_k(b)` when `c` does not mention
    /// `k`. Every element still evaluates the same operations on the same branch.
    ///
    /// A contraction over row-concatenated operands becomes a concatenation of
    /// contractions. The other operand of a binary operator is duplicated into both
    /// branches, which grows the term but not the work any element does.
    HoistSelect,
}

impl Law {
    pub fn exactness(self) -> Exactness {
        match self {
            Law::HoistSelect => Exactness::Exact,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewriteError {
    /// The value is an input or has been eliminated.
    NotDefined { value: ValueId },
}

impl fmt::Display for RewriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RewriteError::NotDefined { value } => write!(f, "value {} has no definition", value.0),
        }
    }
}

impl std::error::Error for RewriteError {}

impl Region {
    /// Replaces every read of `producer` with its definition instantiated at the
    /// read's indices, and returns how many reads were replaced. [`Exactness::Exact`].
    ///
    /// The producer's definition is kept: an output must still be stored, and a
    /// temporary goes only once [`Self::eliminate_dead`] finds it unread. Each
    /// substituted copy gets fresh reduction binders, so none captures an index of
    /// its reader.
    pub fn substitute(&mut self, producer: ValueId) -> Result<usize, RewriteError> {
        let def = self
            .value(producer)
            .and_then(|v| v.definition.clone())
            .ok_or(RewriteError::NotDefined { value: producer })?;
        let names = &mut self.index_names;
        let mut replaced = 0;
        for slot in self.values.iter_mut().skip(producer.0 as usize + 1).flatten() {
            let Some(reader) = &mut slot.definition else {
                continue;
            };
            let body = std::mem::replace(&mut reader.body, Term::Lit(0.0));
            reader.body = body.map_reads(&mut |value, index| {
                if value != producer {
                    return Term::Read { value, index };
                }
                replaced += 1;
                instantiate(&def, &index, names)
            });
        }
        Ok(replaced)
    }

    /// Removes every temporary that no live definition reads, and returns them in
    /// creation order. [`Exactness::Exact`]. Inputs and outputs are never removed.
    pub fn eliminate_dead(&mut self) -> Vec<ValueId> {
        let mut read = vec![false; self.values.len()];
        let mut removed = Vec::new();
        for at in (0..self.values.len()).rev() {
            let Some(value) = &self.values[at] else {
                continue;
            };
            if value.role == Role::Temporary && !read[at] {
                self.values[at] = None;
                removed.push(ValueId(at as u32));
                continue;
            }
            if let Some(def) = &value.definition {
                def.body.for_each_read(&mut |v, _| read[v.0 as usize] = true);
            }
        }
        removed.reverse();
        removed
    }

    /// Substitutes every temporary into its readers, then eliminates them, and returns
    /// how many reads were replaced. [`Exactness::Exact`].
    ///
    /// This is the most aggressive exact plan: nothing scheme-local is stored. It may
    /// duplicate a producer read more than once; choosing what to materialize instead
    /// belongs to the schedule.
    pub fn inline_temporaries(&mut self) -> usize {
        let temporaries: Vec<ValueId> = self
            .values()
            .filter(|(_, v)| v.role == Role::Temporary)
            .map(|(id, _)| id)
            .collect();
        let replaced = temporaries
            .into_iter()
            .map(|id| self.substitute(id).expect("a temporary has a definition"))
            .sum();
        self.eliminate_dead();
        replaced
    }

    /// Rounds the product each definition ends in, through selects, and returns how
    /// many it rounded. [`Exactness::Exact`].
    ///
    /// Run separately, an operation stores its result before the next reads it, so a
    /// device cannot contract one operation's final product into the operation that
    /// reads it. Rounding keeps that true once [`Self::substitute`] joins them.
    pub fn round_products(&mut self) -> usize {
        fn round(term: Term, rounded: &mut usize) -> Term {
            match term {
                Term::Binary { op: BinaryOp::Mul, .. } => {
                    *rounded += 1;
                    Term::unary(UnaryOp::Round, term)
                }
                Term::Select {
                    lhs,
                    cmp,
                    rhs,
                    then,
                    otherwise,
                } => Term::select(lhs, cmp, rhs, round(*then, rounded), round(*otherwise, rounded)),
                other => other,
            }
        }
        let mut rounded = 0;
        for def in self.values.iter_mut().flatten().filter_map(|v| v.definition.as_mut()) {
            let body = std::mem::replace(&mut def.body, Term::Lit(0.0));
            def.body = round(body, &mut rounded);
        }
        rounded
    }

    /// Applies `law` throughout `value`'s definition, and returns how many times it
    /// applied.
    pub fn apply(&mut self, value: ValueId, law: Law) -> Result<usize, RewriteError> {
        let def = self
            .values
            .get_mut(value.0 as usize)
            .and_then(Option::as_mut)
            .and_then(|v| v.definition.as_mut())
            .ok_or(RewriteError::NotDefined { value })?;
        let body = std::mem::replace(&mut def.body, Term::Lit(0.0));
        let mut applied = 0;
        def.body = match law {
            Law::HoistSelect => hoist_select(body, &mut applied),
        };
        Ok(applied)
    }
}

fn instantiate(def: &Definition, args: &[Affine], names: &mut Vec<String>) -> Term {
    let binders: HashMap<IndexVar, IndexVar> = def
        .body
        .binders()
        .into_iter()
        .map(|b| (b, Region::fresh_index(names, b)))
        .collect();
    let domain: HashMap<IndexVar, Affine> = def.domain.iter().copied().zip(args.iter().cloned()).collect();
    def.body.instantiate(&|v| domain.get(&v).cloned(), &binders)
}

fn hoist_select(term: Term, applied: &mut usize) -> Term {
    let term = term.map_children(&mut |c| hoist_select(c, applied));
    let split = |lhs: Affine, cmp, rhs: Affine, then: Term, otherwise: Term, applied: &mut usize| {
        *applied += 1;
        Term::select(
            lhs,
            cmp,
            rhs,
            hoist_select(then, applied),
            hoist_select(otherwise, applied),
        )
    };
    match term {
        Term::Unary { op, arg } => match *arg {
            Term::Select {
                lhs,
                cmp,
                rhs,
                then,
                otherwise,
            } => split(
                lhs,
                cmp,
                rhs,
                Term::unary(op, *then),
                Term::unary(op, *otherwise),
                applied,
            ),
            arg => Term::unary(op, arg),
        },
        Term::Binary { op, lhs, rhs } => match (*lhs, *rhs) {
            (
                Term::Select {
                    lhs: l,
                    cmp,
                    rhs: r,
                    then,
                    otherwise,
                },
                other,
            ) => split(
                l,
                cmp,
                r,
                Term::binary(op, *then, other.clone()),
                Term::binary(op, *otherwise, other),
                applied,
            ),
            (
                other,
                Term::Select {
                    lhs: l,
                    cmp,
                    rhs: r,
                    then,
                    otherwise,
                },
            ) => split(
                l,
                cmp,
                r,
                Term::binary(op, other.clone(), *then),
                Term::binary(op, other, *otherwise),
                applied,
            ),
            (lhs, rhs) => Term::binary(op, lhs, rhs),
        },
        Term::Reduce {
            op,
            index,
            extent,
            order,
            body,
        } => match *body {
            Term::Select {
                lhs,
                cmp,
                rhs,
                then,
                otherwise,
            } if !lhs.mentions(Sym::Index(index)) && !rhs.mentions(Sym::Index(index)) => split(
                lhs,
                cmp,
                rhs,
                Term::reduce_in(op, order, index, extent, *then),
                Term::reduce_in(op, order, index, extent, *otherwise),
                applied,
            ),
            body => Term::reduce_in(op, order, index, extent, body),
        },
        other => other,
    }
}
