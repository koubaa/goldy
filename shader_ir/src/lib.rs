//! Shared shader IR and structured kernel ABI for Goldy frontends.
//!
//! The Rust `#[goldy::compute]` proc-macro lowers a restricted GPU dialect into
//! this IR, emits canonical `[goldy_compute]` Slang, and embeds a [`KernelDef`]
//! for typed Scheme recording. Raw hand-written Slang continues to parse into
//! the same ABI shape via Goldy's virtual-main path.

#![forbid(unsafe_code)]

mod abi;
mod emit;
mod ir;

pub use abi::*;
pub use emit::{emit_canonical_compute_source, emit_user_helper_body};
pub use ir::*;
