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
//! let tv = graph.submit(&ctx)?;
//! context.wait_until(tv)?;
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
//! | [`TaskGraph::write_buffer`] / [`TaskGraph::write_parcel`] | CPU→GPU buffer upload |
//! | [`TaskGraph::copy_render_target_to_swapchain`] | Offscreen render target → swapchain blit |
//!
//! # SWMR scheduling
//!
//! [`NodeAccess`] is orthogonal to a buffer's physical
//! [`BufferKind`](crate::BufferKind). A `Scattered` (read/write) buffer might
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
pub mod record;

pub(crate) use graph::{apply_stamp_targets, IrSubmitState};
pub use graph::{NodeBuilder, RenderPassBuilder, ShaderResourceSlot, TaskGraph};
pub use ir::{BarrierUsage, GraphIR, NodeAccess, NodeAccessUnion, SlotUsageSet, UsageKindFlags};
pub(crate) use ir::{DispatchDim, NodeKind, ResourceBinding, TaskNode};
pub use record::{ComputeNodeRecord, RenderPassRecord};

use crate::backend::{BufferHandle, TextureHandle};
use crate::types::TextureFormat;

/// Opaque id for a [`TaskGraph::transient_buffer`] allocation (graph-scoped bump heap).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransientId(pub u32);

/// Opaque id for a [`TaskGraph::transient_texture`] allocation (graph-scoped transient).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransientTextureId(pub u32);

/// Sentinel value stored in `NodeKind::Dispatch::resource_slots` at the position
/// of a `SwapchainOutput` binding.  Replaced by the real UAV bindless index in
/// `TaskGraph::lower_swapchain_output` after `surface.begin()`.
pub const SWAPCHAIN_SLOT_PLACEHOLDER: u32 = u32::MAX - 1;

/// Opaque handle returned by [`TaskGraph::declare_swapchain_output`].
///
/// Passed to [`NodeBuilder::bind_swapchain_output`] when recording the
/// fine-pass dispatch.  Carries no data — it exists purely for type-safety so
/// callers cannot accidentally swap a concrete texture with a swapchain output.
///
/// The caller (ekrano's `collect_bindless_indices_into`) must place
/// [`SWAPCHAIN_SLOT_PLACEHOLDER`] in the `resource_slots` at the position
/// corresponding to this binding.  `TaskGraph::lower_swapchain_output` then
/// patches that sentinel with the real UAV index after `surface.begin()`.
#[derive(Debug, Clone, Copy)]
pub struct SwapchainOutputHandle;

#[derive(Debug, Clone)]
pub(crate) struct TransientBufferSpec {
    pub(crate) id: u32,
    pub(crate) size: u64,
    /// Element stride for the structured buffer descriptor (bytes).
    /// Defaults to 4 (u32) when not specified.
    pub(crate) stride: u32,
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
    /// Offscreen [`crate::RenderTarget`] color attachment.
    ///
    /// [`NodeKind::RenderPass`] nodes implicitly write their target; consumers
    /// such as [`super::graph::TaskGraph::copy_render_target_to_swapchain`] declare
    /// an explicit `Read` binding so the scheduler orders copy after render.
    RenderTarget(crate::backend::RenderTargetHandle),
    /// Graph-scoped transient; lowered to [`ResourceId::BufferRange`] before submission.
    TransientBuffer(TransientId),
    /// Graph-scoped transient texture; lowered to [`crate::Texture`] before submission.
    TransientTexture(TransientTextureId),
    /// Swapchain output: late-bound at submit time via [`Surface::submit_graph`](crate::Surface::submit_graph).
    ///
    /// Records a stable dependency placeholder without requiring an acquired
    /// swapchain image.  Lowered to [`ResourceId::Texture`] after
    /// `surface.begin()` runs between early and final partition submission.
    SwapchainOutput,
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
            ResourceId::RenderTarget(_) => None,
            ResourceId::TransientBuffer(_) => None,
            ResourceId::TransientTexture(_) => None,
            ResourceId::SwapchainOutput => None,
        }
    }
}

/// Resolved storage for a transient buffer slot.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct ResolvedTransientBuffer {
    pub parent: BufferHandle,
    pub offset: u64,
    pub len: u64,
    pub uav_index: u32,
    pub srv_index: u32,
}

/// Resolved storage for a transient texture slot.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct ResolvedTransientTexture {
    pub handle: TextureHandle,
}

/// Resolved storage for the swapchain output slot.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct ResolvedSwapchain {
    pub handle: TextureHandle,
    pub uav_index: u32,
}

/// Maps promised slots to their concrete storage for the current submission.
///
/// The IR is invariant data written by the user; the runtime resolves
/// promised slots (`TransientBuffer`, `TransientTexture`, `SwapchainOutput`)
/// through this table at emission time.  No IR clone is ever necessary.
///
/// Transient entries are page-local (supplied by `PlacementHeap::advance_page`).
/// The swapchain entry is boundary-local (filled after `surface.begin()`).
#[derive(Debug, Clone, Default)]
pub(crate) struct SlotResolver {
    pub buffers: std::collections::HashMap<u32, ResolvedTransientBuffer>,
    pub textures: std::collections::HashMap<u32, ResolvedTransientTexture>,
    pub swapchain: Option<ResolvedSwapchain>,
}

impl SlotResolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve a `ResourceId` to its concrete form.  Concrete ids pass through.
    #[allow(dead_code)]
    pub fn resolve(&self, id: ResourceId) -> ResourceId {
        match id {
            ResourceId::TransientBuffer(t) => {
                let r = &self.buffers[&t.0];
                ResourceId::BufferRange {
                    parent: r.parent,
                    offset: r.offset,
                    len: r.len,
                }
            }
            ResourceId::TransientTexture(t) => ResourceId::Texture(self.textures[&t.0].handle),
            ResourceId::SwapchainOutput => {
                let sc = self
                    .swapchain
                    .as_ref()
                    .expect("SlotResolver::resolve: SwapchainOutput accessed before swapchain acquired");
                ResourceId::Texture(sc.handle)
            }
            other => other,
        }
    }

    /// Resolve a dispatch's `resource_slots`, patching transient and swapchain
    /// entries to their concrete bindless indices.
    pub fn resolve_slots(&self, resource_slots: &[u32], bindings: &[ir::ResourceBinding]) -> Vec<u32> {
        let mut out = resource_slots.to_vec();
        for (i, b) in bindings.iter().enumerate() {
            if i >= out.len() {
                break;
            }
            match b.resource {
                ResourceId::TransientBuffer(t) => {
                    let r = &self.buffers[&t.0];
                    let is_read_only = b.access == ir::NodeAccess::Read;
                    out[i] = if is_read_only { r.srv_index } else { r.uav_index };
                }
                ResourceId::SwapchainOutput if out[i] == SWAPCHAIN_SLOT_PLACEHOLDER => {
                    let sc = self
                        .swapchain
                        .as_ref()
                        .expect("SlotResolver::resolve_slots: SwapchainOutput before acquire");
                    out[i] = sc.uav_index;
                }
                _ => {}
            }
        }
        out
    }
}
