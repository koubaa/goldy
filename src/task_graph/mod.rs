//! Shared graph IR, barrier analysis, and submit engine used by [`crate::Scheme`].
//!
//! Schemes record a [`GraphIR`], then submit through the IR submit path, which
//! schedules waves, inserts barriers, and drives backend `submit_graph` paths.

pub(crate) mod analysis;
pub(crate) mod cb_replay;
pub(crate) mod cross_submit;
pub(crate) use cross_submit::CrossSubmitSync;
mod graph;
mod ir;
pub mod record;

pub use graph::ShaderResourceSlot;
pub(crate) use graph::{DeferredPresentAcquire, IrSubmitState, ResolvedPresentSlot};
pub use ir::{BarrierSet, BarrierUsage, GraphIR, NodeAccess, NodeAccessUnion, SlotUsageSet, UsageKindFlags};
pub(crate) use ir::{DispatchDim, NodeKind, ResourceBinding, TaskNode};
pub use record::{ComputeNodeRecord, RenderPassRecord};

use crate::backend::{BufferHandle, TextureHandle};
use crate::types::TextureFormat;

/// Opaque id for a transient buffer slot in graph IR (`ResourceId::TransientBuffer`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TransientId(pub u32);

/// Opaque id for a transient texture slot in graph IR (`ResourceId::TransientTexture`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransientTextureId(pub u32);

/// Sentinel value stored in `NodeKind::Dispatch::resource_slots` at the position
/// of a `SwapchainOutput` binding. Replaced by the real UAV bindless index when
/// a surface resolves swapchain backing at submit time.
pub const SWAPCHAIN_SLOT_PLACEHOLDER: u32 = u32::MAX - 1;

/// Sentinel value stored in `NodeKind::Dispatch::resource_slots` at the position
/// of a `ResourceId::PresentLease` binding. Replaced by the real UAV bindless
/// index when the swapchain pool resolves backing at submit time.
pub const PRESENT_LEASE_SLOT_PLACEHOLDER: u32 = u32::MAX - 2;

/// Opaque handle for a swapchain output binding in graph IR.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct SwapchainOutputHandle;

#[derive(Debug, Clone)]
#[allow(dead_code)]
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
#[allow(dead_code)]
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
    /// such as scheme copy-to-present nodes declare
    /// an explicit `Read` binding so the scheduler orders copy after render.
    RenderTarget(crate::backend::RenderTargetHandle),
    /// Graph-scoped transient; lowered to [`ResourceId::BufferRange`] before submission.
    #[allow(dead_code)] // TaskGraph placement-heap path removed; kept for IR/analysis matching
    TransientBuffer(TransientId),
    /// Graph-scoped transient texture; lowered to [`crate::Texture`] before submission.
    #[allow(dead_code)]
    TransientTexture(TransientTextureId),
    /// Swapchain output: late-bound at submit time (legacy surface-graph path).
    ///
    /// Records a stable dependency placeholder without requiring an acquired
    /// swapchain image.  Lowered to [`ResourceId::Texture`] after acquire.
    #[allow(dead_code)]
    SwapchainOutput,
    /// Present binding: scheme-unique id for a swapchain-pool drawable.
    ///
    /// The `u32` is allocated by [`crate::Scheme`] when a [`crate::PresentLease`]
    /// is first recorded; it is **not** the pool-local lease id. Two pools that
    /// both expose local id `0` receive distinct binding ids in one scheme.
    ///
    /// Lowered to [`ResourceId::Texture`] when the pool acquires a backing slot
    /// at [`crate::Scheme::submit`] time.
    PresentLease(u32),
    /// Scheme-scoped logical upload buffer: late-bound CPU-writable staging parcel.
    ///
    /// Declared via [`crate::Scheme::declare_upload_buffer`]; physical backing is
    /// selected by [`crate::Scheme::stage_upload_buffer`] and resolved at submit
    /// through [`SlotResolver::upload_buffers`].
    UploadBuffer(u32),
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
            ResourceId::PresentLease(_) => None,
            ResourceId::UploadBuffer(_) => None,
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

/// Resolved physical staging parcel for a scheme [`ResourceId::UploadBuffer`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedUploadBuffer {
    pub parent: BufferHandle,
    pub offset: u64,
    pub len: u64,
}

/// Maps promised slots to their concrete storage for the current submission.
///
/// The IR is invariant data written by the user; the runtime resolves
/// promised slots (`TransientBuffer`, `TransientTexture`, `SwapchainOutput`,
/// `PresentLease`, `UploadBuffer`) through this table at emission time.
/// No IR clone is ever necessary.
///
/// Transient entries are page-local (supplied by `PlacementHeap::advance_page`).
/// The swapchain entry is boundary-local (filled after `surface.begin()`).
/// Upload buffers are scheme-local (filled by [`crate::Scheme`] before submit).
#[derive(Debug, Clone, Default)]
pub(crate) struct SlotResolver {
    pub buffers: std::collections::HashMap<u32, ResolvedTransientBuffer>,
    pub textures: std::collections::HashMap<u32, ResolvedTransientTexture>,
    pub swapchain: Option<ResolvedSwapchain>,
    /// Per [`ResourceId::PresentLease`] id, resolved at scheme submit time.
    pub present_leases: std::collections::HashMap<u32, ResolvedSwapchain>,
    /// Per [`ResourceId::UploadBuffer`] id, resolved at scheme submit time.
    pub upload_buffers: std::collections::HashMap<u32, ResolvedUploadBuffer>,
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
            ResourceId::PresentLease(id) => {
                let sc = self
                    .present_leases
                    .get(&id)
                    .expect("SlotResolver::resolve: PresentLease accessed before pool acquire");
                ResourceId::Texture(sc.handle)
            }
            ResourceId::UploadBuffer(id) => {
                let u = self
                    .upload_buffers
                    .get(&id)
                    .expect("SlotResolver::resolve: UploadBuffer accessed before stage/submit resolve");
                ResourceId::BufferRange {
                    parent: u.parent,
                    offset: u.offset,
                    len: u.len,
                }
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
                ResourceId::PresentLease(id) if out[i] == PRESENT_LEASE_SLOT_PLACEHOLDER => {
                    let sc = self
                        .present_leases
                        .get(&id)
                        .expect("SlotResolver::resolve_slots: PresentLease before pool acquire");
                    out[i] = sc.uav_index;
                }
                _ => {}
            }
        }
        out
    }
}
