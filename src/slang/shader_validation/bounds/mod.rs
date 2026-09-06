//! Static bounds analysis over Slang IR (prototype).
//!
//! Slang already rejects *constant* out-of-bounds indices, but a dynamic index such as
//! `links[link]` is only checked at runtime — and on most GPUs "checked" means an undefined
//! read, a hang, or a device loss. This module proves, conservatively, that every dynamic
//! index into a statically sized array (`groupshared`, globals, locals, struct members,
//! vectors, matrices) stays inside `0 <= index < length`, and reports every access it could
//! *not* prove, with the Slang source location and the call path that reaches it.
//!
//! One rule of [`crate::slang::shader_validation`] (`GOLDY_SHADER_VALIDATION=bounds`); it
//! never fails a compile, findings are warnings. The input is a linked `Module` from the
//! shared IR toolkit ([`crate::slang::ir`]): the translation unit plus
//! the modules it imports, with debug info so findings carry source locations. See
//! `docs/src/design/shader-bounds-analysis.md` for the integration decision (Slang IR vs
//! SPIR-V), the analysis model, and known false positives.
//!
//! # Analysis model
//!
//! 1. **Values.** Every integer scalar/vector gets an interval per lane; structs are tracked
//!    field by field; everything else is opaque. Local `var`s that never escape are read
//!    through their stores, so the `var`/`store`/`load` pattern Slang uses for constructors
//!    and `out` parameters does not lose information.
//! 2. **Interval propagation.** A flow-insensitive ascending fixpoint with widening followed
//!    by narrowing over each function. Block parameters (Slang's phis) join their incoming
//!    values, each evaluated under the facts that hold on its edge, so the back edge of
//!    `for (i = 0; i < n; i++)` contributes `[1, n]` rather than a wrapped increment.
//! 3. **Path-sensitive refinement.** For each dynamic index the dominator tree is walked for
//!    `ifElse`/`conditionalBranch`/`switch` conditions that dominate the access; the index
//!    expression is re-evaluated under those facts, including the relational rule
//!    `a >= b  ==>  a - b in [0, hi(a) - lo(b)]` that workgroup scans rely on.
//! 4. **Interprocedural.** Calls to functions with bodies (in this module or an imported one,
//!    generics included) are analyzed in the calling context: the callee's parameters take the
//!    argument intervals at the call site, and its return value flows back. Summaries are
//!    memoized per `(function, argument intervals)`; recursion and depth are bounded.
//!    Core-module intrinsics (`min`, `clamp`, `WaveGetLaneCount`, ...) are modeled by name.
//! 5. **Check.** Each `getElementPtr`/`getElement` index into a known-length aggregate must
//!    satisfy `0 <= index <= length - 1`. Anything else is a [`BoundsDiagnostic`], reported once
//!    per access with the union of the index ranges over every context that reaches it.

use std::fmt;

use crate::slang::ir::{Module, SourceLocation};

mod analysis;

/// One dynamic array access the analysis could not prove in bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundsDiagnostic {
    /// Name of the function containing the access.
    pub function: String,
    /// Functions on the shortest call path from an entry point to `function` (the entry point
    /// first, `function` itself excluded). Empty when the access is in the entry point.
    pub call_path: Vec<String>,
    /// Name of the indexed array (variable name plus struct member path when available).
    pub array: String,
    /// Statically known number of elements.
    pub array_length: u64,
    /// Interval the index was proven to lie in (in the index type's signedness), or `None`
    /// when nothing narrower than the full type range is known. When several calling
    /// contexts reach the access this is the union over all of them.
    pub index_range: Option<(i128, i128)>,
    /// Slang source location, when the module carries debug info.
    pub location: Option<SourceLocation>,
    /// What the index ultimately depends on that the analysis cannot bound (system values,
    /// buffer loads, float conversions, functions without a body, ...). Empty when the range
    /// is known but simply too wide. Deduplicated, at most a few entries.
    pub depends_on: Vec<String>,
}

impl fmt::Display for BoundsDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "possible out-of-bounds index into `{}[{}]`",
            self.array, self.array_length
        )?;
        match self.index_range {
            Some((lo, hi)) => {
                write!(f, ": index range [{lo}, {hi}]")?;
                if lo < 0 {
                    write!(f, " (may be negative)")?;
                }
                if hi >= self.array_length as i128 {
                    write!(f, " (may exceed {})", self.array_length as i128 - 1)?;
                }
            }
            None => write!(f, ": index range unknown")?,
        }
        if !self.depends_on.is_empty() {
            write!(f, " (depends on {})", self.depends_on.join(", "))?;
        }
        if let Some(loc) = &self.location {
            write!(f, " at {loc}")?;
        }
        write!(f, " in `{}`", self.function)?;
        if !self.call_path.is_empty() {
            write!(f, " (called from {})", self.call_path.join(" -> "))?;
        }
        Ok(())
    }
}

/// Result of the bounds check over one shader (see
/// [`validate`](crate::slang::shader_validation::validate)).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BoundsReport {
    /// Accesses that could not be proven safe.
    pub diagnostics: Vec<BoundsDiagnostic>,
    /// Number of dynamic (non-constant) indices into known-length aggregates that were
    /// checked (each access counts once however many contexts reach it).
    pub checked_accesses: usize,
    /// Number of those proven to be in bounds in every context.
    pub proven_safe: usize,
}

impl BoundsReport {
    /// `true` when every checked access was proven in bounds.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

/// Check every dynamic index in `module` (a translation unit linked with its imports).
///
/// Entry points and, when the translation unit has none, its exported functions are the
/// analysis roots; everything reachable through calls is analyzed in the calling context.
#[must_use]
pub(crate) fn analyze(module: &Module) -> BoundsReport {
    analysis::Analyzer::new(module).run()
}

#[cfg(test)]
mod tests;
