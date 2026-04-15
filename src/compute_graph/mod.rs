//! Compute graph API for explicit GPU dispatch scheduling.
//!
//! # Motivation
//!
//! Goldy's bindless model (heap-backed argument buffers, push-constant indices)
//! gives shaders flexible, low-overhead access to resources. However, it makes
//! the GPU's automatic dependency tracking blind — Metal cannot see through
//! argument buffer indirection to know which resources a dispatch reads or
//! writes, so it cannot insert barriers automatically.
//!
//! The current workaround (one command buffer per dispatch) is correct but
//! suboptimal: each command buffer carries scheduling overhead, and Metal
//! serializes them within a queue, preventing independent dispatches from
//! overlapping.
//!
//! This module pairs bindless **access** with explicit **scheduling** — a
//! compute graph that declares what each dispatch reads and writes, so Goldy
//! can insert minimal barriers and maximize parallelism on all backends.
//!
//! # Design: two-tier API
//!
//! Both tiers share one underlying graph IR. They are opt-in alongside the
//! existing [`ComputeEncoder`](crate::ComputeEncoder).
//!
//! ## Tier 1: [`ComputeGraph`] — interpreted, dynamic
//!
//! Build a DAG of dispatch nodes with per-resource access declarations each
//! frame. At submit time Goldy analyzes the graph, inserts barriers, and
//! executes. Best for dynamic workloads or prototyping.
//!
//! ```rust,ignore
//! let mut graph = ComputeGraph::new();
//!
//! graph.node("pathtag_reduce", &pipeline_a)
//!     .bind_buffer(&scene_buf, NodeAccess::Read)
//!     .bind_buffer(&tagmonoid_buf, NodeAccess::ReadWrite)
//!     .push_constants_raw(&[scene_idx, tagmonoid_idx])
//!     .dispatch(64, 1, 1);
//!
//! graph.node("bbox_clear", &pipeline_b)
//!     .bind_buffer(&bbox_buf, NodeAccess::Write)      // independent of above
//!     .push_constants_raw(&[bbox_idx])
//!     .dispatch(16, 1, 1);
//!
//! graph.submit(&device)?.wait()?;
//! ```
//!
//! ## Tier 2: [`ComputeProgram`] — compiled, specializable
//!
//! Separate graph topology (static) from bindings and dimensions (dynamic).
//! Compile once, specialize cheaply per frame. Analogous to NVIDIA Warp's JIT
//! model where a graph is compiled and then specialized with runtime values.
//!
//! ```rust,ignore
//! // Build phase (once)
//! let mut builder = ComputeProgram::builder();
//! let scene     = builder.buffer_slot("scene");
//! let tagmonoid = builder.buffer_slot("tagmonoid");
//! let wg_reduce = builder.dim_slot("wg_reduce");
//!
//! builder.step("pathtag_reduce", &pipeline_a)
//!     .bind_buffer(scene, NodeAccess::Read)
//!     .bind_buffer(tagmonoid, NodeAccess::ReadWrite)
//!     .dispatch_slot(wg_reduce);
//!
//! let program = builder.compile()?;
//!
//! // Execute phase (each frame)
//! let mut exec = program.specialize();
//! exec.bind_buffer(scene, &scene_buf);
//! exec.bind_buffer(tagmonoid, &tagmonoid_buf);
//! exec.set_dim(wg_reduce, (64, 1, 1));
//! exec.submit(&device)?.wait()?;
//! ```
//!
//! # SWMR scheduling
//!
//! [`NodeAccess`] is orthogonal to a buffer's physical
//! [`DataAccess`](crate::DataAccess). A `Scattered` (read/write) buffer might
//! be read-only in one dispatch and read-write in another. The graph uses
//! per-node logical access to enable single-writer/multiple-reader parallelism:
//!
//! - Multiple `Read` nodes on the same resource run concurrently.
//! - A `Write` or `ReadWrite` node serializes against all prior accessors.
//! - Barriers are inserted only at true RAW/WAR/WAW edges.
//!
//! # Backend mapping
//!
//! The graph emits [`ComputeCommand::ResourceBarrier`](crate::backend::ComputeCommand)
//! with per-resource granularity. Each backend handles it:
//!
//! - **Metal**: `memoryBarrierWithResources:count:` — precise per-resource
//!   barriers within a single compute encoder.
//! - **Vulkan**: falls back to global compute pipeline barrier (per-resource
//!   `VkBufferMemoryBarrier` is a future optimization).
//! - **DX12**: falls back to global UAV barrier (per-resource
//!   `D3D12_RESOURCE_BARRIER` is a future optimization).
//!
//! See `docu/research/technical_stack/abstract-gpu-compute-graph.md` for the
//! full design rationale.

mod analysis;
mod graph;
mod ir;
mod program;

pub use graph::{ComputeGraph, NodeBuilder};
pub use ir::{DispatchKind, NodeAccess};
pub use program::{ComputeProgram, DimSlotId, Execution, ProgramBuilder, SlotId, StepBuilder};

use crate::backend::{BufferHandle, TextureHandle};

/// Identifies a GPU resource within a compute graph.
///
/// Used internally by the graph IR. The public API accepts `&Buffer` /
/// `&Texture` and extracts handles automatically (matching the pattern
/// used by [`ComputePass`](crate::ComputePass)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceId {
    Buffer(BufferHandle),
    Texture(TextureHandle),
}
