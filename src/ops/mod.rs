//! Semantic GPU operations recorded on a [`crate::Scheme`].
//!
//! These nodes describe *what* to compute. Each backend chooses *how* on first
//! submit — a vendor library when one exists, otherwise Goldy's portable stdlib
//! kernel — and retains the realized plan for later replays.

pub mod matmul;
pub(crate) mod matmul_kernel;

pub use matmul::{MatMulBuilder, MatMulDType, MatMulDesc, MatMulView};
