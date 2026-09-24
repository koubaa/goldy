//! Symbolic index notation for semantic tensor fusion.
//!
//! A [`Region`] is a set of tensor values. Inputs are read from parcels through affine
//! [`Storage`] maps. Temporaries and outputs are defined in index notation:
//!
//! ```text
//! y[i]  = sum{k<288}(W[i, k] * h[k])
//! x2[i] = x[i] + y[i]
//! ```
//!
//! Accesses are integer [`Affine`] functions of index variables and index parameters.
//! Scalar [`Term`]s are element-wise arithmetic, affine selects and reductions. A term's
//! tree is its evaluation order, because floating-point arithmetic is not associative.
//!
//! Fusion is algebra over definitions. [`Region::substitute`] places a producer's
//! definition in its readers, [`Region::eliminate_dead`] drops unread temporaries, and
//! [`Region::apply`] applies a local [`Law`]. Every rewrite states the [`Exactness`] it
//! preserves. [`Region::evaluate`] is the reference semantics that exactness is
//! measured against.
//!
//! Regions compose: [`Region::then`] appends a region that runs afterwards, turning
//! reads of storage the first one writes into reads of its values.
//!
//! The notation is device-independent. Choosing what to materialize and how to map
//! indices onto a grid is a [`Schedule`], which is separate; [`lower`] realizes a region
//! under one as a single kernel.

mod affine;
mod compose;
mod interp;
mod region;
mod rewrite;
mod schedule;
mod term;

#[cfg(test)]
mod compose_tests;
#[cfg(test)]
mod tests;

pub use affine::{Affine, IndexParam, IndexVar, Sym};
pub use compose::{Appended, ComposeError};
pub use interp::{Environment, EvalError};
pub use region::{Definition, ParcelId, Region, RegionError, Role, Storage, Value};
pub use rewrite::{Exactness, Law, RewriteError};
pub use schedule::{
    lower, lower_with, IndexSource, LowerError, Lowered, Schedule, LANE_WORKGROUP_THREADS, WORKGROUP_THREADS,
};
pub use term::{BinaryOp, CmpOp, Invariant, ReduceOp, ReduceOrder, ScalarParam, Term, UnaryOp, ValueId};
