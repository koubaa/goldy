//! What a lowered kernel does, counted in units a device model prices.
//!
//! Lowering counts what each thread of its kernel executes and what the kernel moves
//! through storage; it does not know how fast any device is. A caller that compares
//! plans, such as a fusion planner choosing between a fused kernel and the dispatches
//! it replaces, turns [`Estimate`]s into time with its own device model.

use super::term::{Term, UnaryOp};

/// Device-independent counts for one kernel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Estimate {
    /// Distinct bytes of storage the kernel reads or writes.
    pub bytes: u64,
    /// Dependent steps of its longest thread: operations, with every iteration of every
    /// loop around them counted.
    pub serial: u64,
    /// Storage reads among those steps that wait on one another, such as one per
    /// iteration of a loop that reads storage. A GPU thread does not run ahead of a
    /// read it needs, so each costs a round trip to memory.
    pub loads: u64,
    /// Operations over all of its threads.
    pub work: u64,
    /// Multiply-adds it runs on matrix units.
    pub matrix: u64,
}

/// Operations one evaluation of `term` executes; subterms alpha-equal to one of
/// `known` are already in registers and cost nothing.
pub(super) fn ops(term: &Term, known: &[&Term]) -> u64 {
    if is_known(term, known) {
        return 0;
    }
    match term {
        Term::Lit(_) | Term::Scalar(_) | Term::IndexValue(_) => 0,
        Term::Read { .. } => 1,
        Term::Unary {
            op: UnaryOp::Round,
            arg,
        } => ops(arg, known),
        Term::Unary { arg, .. } => 1 + ops(arg, known),
        Term::Binary { lhs, rhs, .. } => 1 + ops(lhs, known) + ops(rhs, known),
        Term::Select { then, otherwise, .. } => 1 + ops(then, known).max(ops(otherwise, known)),
        Term::Reduce { extent, body, .. } => u64::from(*extent) * (1 + ops(body, known)),
    }
}

/// Dependent steps of one evaluation of `term`: operands evaluate independently, and
/// a reduction's iterations one after another.
pub(super) fn serial(term: &Term, known: &[&Term]) -> u64 {
    if is_known(term, known) {
        return 0;
    }
    match term {
        Term::Lit(_) | Term::Scalar(_) | Term::IndexValue(_) => 0,
        Term::Read { .. } => 1,
        Term::Unary {
            op: UnaryOp::Round,
            arg,
        } => serial(arg, known),
        Term::Unary { arg, .. } => 1 + serial(arg, known),
        Term::Binary { lhs, rhs, .. } => 1 + serial(lhs, known).max(serial(rhs, known)),
        Term::Select { then, otherwise, .. } => 1 + serial(then, known).max(serial(otherwise, known)),
        Term::Reduce { extent, body, .. } => u64::from(*extent) * (1 + serial(body, known)),
    }
}

/// Storage reads of one evaluation of `term` that wait on one another: operands are
/// read together, and a reduction's iterations one after another.
pub(super) fn loads(term: &Term, known: &[&Term]) -> u64 {
    if is_known(term, known) {
        return 0;
    }
    match term {
        Term::Lit(_) | Term::Scalar(_) | Term::IndexValue(_) => 0,
        Term::Read { .. } => 1,
        Term::Unary { arg, .. } => loads(arg, known),
        Term::Binary { lhs, rhs, .. } => loads(lhs, known).max(loads(rhs, known)),
        Term::Select { then, otherwise, .. } => loads(then, known).max(loads(otherwise, known)),
        Term::Reduce { extent, body, .. } => u64::from(*extent) * loads(body, known),
    }
}

fn is_known(term: &Term, known: &[&Term]) -> bool {
    !matches!(
        term,
        Term::Lit(_) | Term::Scalar(_) | Term::IndexValue(_) | Term::Read { .. }
    ) && known.iter().any(|k| k.alpha_eq(term))
}
