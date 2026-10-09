//! Shared shader IR and structured kernel ABI for Goldy frontends.
//!
//! The Rust `#[goldy::compute]` proc-macro lowers a restricted GPU dialect into
//! this IR, emits canonical `[goldy_compute]` Slang, and embeds a [`KernelDef`]
//! for typed Scheme recording. The [`ShaderKernel`] definition is retained beside
//! that source so it can be lowered on its own or composed with other definitions.
//! Raw hand-written Slang continues to parse into the same ABI shape via Goldy's
//! virtual-main path.
//!
//! [`algebra`] describes tensor computations in symbolic index notation, for fusions
//! derived from the mathematics of an operation rather than from its kernel body.

#![forbid(unsafe_code)]

mod abi;
pub mod algebra;
mod emit;
mod forward;
mod fuse;
mod ir;
mod symbols;

pub use abi::*;
pub use emit::{
    assemble_virtual_entry, emit_canonical_compute_source, emit_user_helper_body, lower_body, tensor_slot_map, BodyEnv,
    LoweredBody, TensorSlot, TensorSlots, VirtualEntrySignature, SUBGROUP_WIDTH_DEFINE, VIRTUAL_ENTRY_NAME,
};
pub use fuse::{
    compose, FusedDefinition, FusedStage, FusionLimits, FusionRejection, FusionStage, ScalarOrigin, FUSION_ABI_VERSION,
    PORTABLE_WORKGROUP_BYTES,
};
pub use ir::*;
pub use symbols::SymbolKind;
