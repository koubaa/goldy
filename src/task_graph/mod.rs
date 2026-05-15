//! Task graph API for explicit GPU scheduling with automatic barrier insertion.
//!
//! # Motivation
//!
//! Goldy's bindless model (heap-backed argument buffers, resource-slot indices)
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
//! task graph that declares what each node reads and writes, so Goldy
//! can insert minimal barriers and maximize parallelism on all backends.
//!
//! # Usage
//!
//! Build a DAG of task nodes with per-resource access declarations, then
//! submit. Goldy analyzes the graph, inserts minimal barriers, and executes
//! with maximum parallelism.
//!
//! ```rust,ignore
//! let mut graph = TaskGraph::new();
//!
//! // Clears and uploads are first-class nodes — the analyzer inserts the
//! // correct barrier between this clear and any downstream reader.
//! graph.clear_buffer(&pool_backing, 0, pool.capacity());
//!
//! graph.node("pathtag_reduce", &pipeline_a)
//!     .bind_buffer(&scene_buf, NodeAccess::Read)
//!     .bind_buffer(&tagmonoid_buf, NodeAccess::ReadWrite)
//!     .bind_resources_raw_slice(&[scene_idx, tagmonoid_idx])
//!     .dispatch(64, 1, 1);
//!
//! graph.node("bbox_clear", &pipeline_b)
//!     .bind_buffer(&bbox_buf, NodeAccess::Write)      // independent of above
//!     .bind_resources_raw_slice(&[bbox_idx])
//!     .dispatch(16, 1, 1);
//!
//! let tv = graph.submit(&device)?;
//! device.wait_until(tv)?;
//! ```
//!
//! # Node kinds
//!
//! A [`TaskGraph`] accepts three types of nodes, all subject to the same
//! dependency analysis:
//!
//! | Builder method            | GPU operation                      |
//! |---------------------------|------------------------------------|
//! | [`TaskGraph::node`]       | Compute dispatch (direct/indirect) |
//! | [`TaskGraph::clear_buffer`] / [`TaskGraph::clear_buffer_view`] | GPU-side buffer zero-fill |
//! | [`TaskGraph::write_buffer`] | CPU→GPU buffer upload              |
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

pub(crate) mod analysis;
mod graph;
mod ir;

pub use graph::{NodeBuilder, RenderPassBuilder, TaskGraph};
pub use ir::{GraphIR, NodeAccess};

use crate::backend::{BufferHandle, TextureHandle};
use crate::types::TextureFormat;

/// Opaque id for a [`TaskGraph::transient_buffer`] allocation (graph-scoped bump heap).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransientId(pub u32);

/// Opaque id for a [`TaskGraph::transient_texture`] allocation (graph-scoped transient).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransientTextureId(pub u32);

#[derive(Debug, Clone)]
pub struct TransientBufferSpec {
    pub id: u32,
    pub size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TransientTextureKey {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
}

#[derive(Debug, Clone)]
pub(crate) struct TransientTextureSpec {
    pub id: u32,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
}

/// Identifies a GPU resource within a task graph.
///
/// Used internally by the graph IR. The public API accepts `&Buffer` /
/// `&BufferView` / `&Texture` and extracts handles automatically.
///
/// # View semantics
///
/// `BufferRange` represents a sub-range of a parent `Buffer` (e.g. a
/// `BufferPool` allocation). Two `BufferRange`s with the same parent are
/// **independent** unless their byte ranges overlap. This enables the
/// scheduler to execute dispatches that touch non-overlapping pool views
/// in the same wave, matching the behaviour of per-allocation tracking in
/// wgpu and the GPU's own memory model.
///
/// Backends never see `BufferRange` — barriers are always emitted against
/// the parent handle via [`ResourceId::canonical_buffer_handle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ResourceId {
    /// A whole-buffer resource (own allocation).
    Buffer(BufferHandle),
    /// A sub-range `[offset, offset+len)` within a parent buffer.
    BufferRange {
        parent: BufferHandle,
        offset: u64,
        len: u64,
    },
    Texture(TextureHandle),
    /// Graph-scoped transient; lowered to [`ResourceId::BufferRange`] before submission.
    TransientBuffer(TransientId),
    /// Graph-scoped transient texture; lowered to [`crate::Texture`] before submission.
    TransientTexture(TransientTextureId),
}

impl ResourceId {
    /// The `BufferHandle` used in backend barrier commands.
    ///
    /// For `Buffer(h)` returns `h`; for `BufferRange { parent, .. }` returns `parent`.
    /// Returns `None` for textures.
    pub(crate) fn canonical_buffer_handle(self) -> Option<BufferHandle> {
        match self {
            ResourceId::Buffer(h) => Some(h),
            ResourceId::BufferRange { parent, .. } => Some(parent),
            ResourceId::Texture(_) => None,
            ResourceId::TransientBuffer(_) => None,
            ResourceId::TransientTexture(_) => None,
        }
    }
}
