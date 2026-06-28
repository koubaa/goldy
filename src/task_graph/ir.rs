//! Task graph intermediate representation: nodes, bindings, and compiled schedules.
//!
//! These types form the shared IR consumed by [`analysis`](super::analysis).
//! The [`analysis`](super::analysis) module consumes a [`GraphIR`] and produces a
//! [`CompiledSchedule`] of [`Wave`]s with `BarrierSet`s.
//!
//! A [`TaskNode`] may be a compute dispatch, a buffer clear, or a buffer write.
//! The analyzer operates only on [`TaskNode::bindings`] and is node-kind-agnostic;
//! [`emit_commands`](super::analysis::emit_commands) switches on [`NodeKind`] to
//! produce the final [`crate::backend::GpuCommand`] stream.

use super::ResourceId;
use crate::backend::{BufferHandle, ComputePipelineHandle, RenderTargetHandle, TextureHandle};
use std::sync::Arc;

bitflags::bitflags! {
    /// Which GPU operation categories are active on a side of a barrier.
    ///
    /// The Koubaa machine distinguishes three dispatch categories that map to
    /// distinct hardware pipeline stages on all supported backends:
    /// - `COMPUTE` — shader dispatches (compute pipeline)
    /// - `TRANSFER` — clears, host-uploads, and GPU-side copies (DMA / copy engine)
    /// - `RENDER` — offscreen render passes (graphics pipeline)
    ///
    /// Flags are unioned across all edges contributing to a given `BarrierSet`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct UsageKindFlags: u8 {
        const COMPUTE  = 0b001;
        const TRANSFER = 0b010;
        const RENDER   = 0b100;
    }
}

/// Access semantics on one side of a barrier edge, in Koubaa-level terms.
///
/// Pairs with [`NodeAccess`] to fully describe what a wave did (src) or will do
/// (dst) to the slots covered by a `BarrierSet`.  Each backend lowers this to
/// its native synchronization primitives (DX12 enhanced-barrier sync/access flags,
/// Vulkan `VkPipelineStageFlags2` / `VkAccessFlags2`, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SlotUsageSet {
    /// Union of read/write access across all contributing edges.
    ///
    /// `Write` or `ReadWrite` means a UAV / storage write was in flight;
    /// `Read` means only shader reads.  Backends use this to narrow the
    /// `AccessBefore` / `AccessAfter` flags.
    pub access: NodeAccessUnion,
    /// Which pipeline categories (compute / transfer / render) contributed.
    pub kinds: UsageKindFlags,
}

/// The most permissive read/write access seen across a set of edges.
///
/// Distinct from [`NodeAccess`] (which is per-binding) because a wave may
/// contain both read-only and read-write bindings to the same resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NodeAccessUnion {
    /// Only reads observed — no storage write in flight.
    #[default]
    ReadOnly,
    /// At least one write (or read-write) observed.
    Write,
}

impl NodeAccessUnion {
    pub fn widen(self, access: NodeAccess) -> Self {
        if access.writes() {
            Self::Write
        } else {
            self
        }
    }
    pub fn writes(self) -> bool {
        matches!(self, Self::Write)
    }
}

impl SlotUsageSet {
    /// Absorb the access/kind of one more node into this set.
    pub fn merge(&mut self, access: NodeAccess, kind: UsageKindFlags) {
        self.access = self.access.widen(access);
        self.kinds |= kind;
    }

    pub fn is_empty(self) -> bool {
        self.kinds.is_empty()
    }
}

/// Logical access a task node has on a resource, orthogonal to the
/// resource's physical [`BufferKind`](crate::BufferKind) /
/// [`TextureKind`](crate::TextureKind).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeAccess {
    /// Node only reads — can overlap with other `Read` nodes on the same resource.
    Read,
    /// Node only writes — requires exclusive access.
    Write,
    /// Node reads and writes — requires exclusive access.
    ReadWrite,
}

impl NodeAccess {
    pub fn writes(self) -> bool {
        matches!(self, NodeAccess::Write | NodeAccess::ReadWrite)
    }

    pub fn reads(self) -> bool {
        matches!(self, NodeAccess::Read | NodeAccess::ReadWrite)
    }
}

impl From<crate::types::ResourceAccess> for NodeAccess {
    fn from(access: crate::types::ResourceAccess) -> Self {
        match access {
            crate::types::ResourceAccess::Read => NodeAccess::Read,
            crate::types::ResourceAccess::Write => NodeAccess::Write,
            crate::types::ResourceAccess::ReadWrite => NodeAccess::ReadWrite,
        }
    }
}

/// A single resource binding within a task node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResourceBinding {
    /// Graph IR only; not exposed publicly so [`super::ResourceId`] can stay crate-private.
    pub(crate) resource: ResourceId,
    pub access: NodeAccess,
}

/// Dispatch dimensions for a compute task node.
#[derive(Debug, Clone)]
pub enum DispatchDim {
    /// Fixed workgroup counts known at graph construction time.
    Direct { x: u32, y: u32, z: u32 },
    /// Workgroup counts read from a buffer at runtime (3× `u32` at `offset`).
    Indirect { buffer: BufferHandle, offset: u64 },
}

/// What a task node actually does on the GPU.
///
/// The analyzer only looks at [`TaskNode::bindings`]; `NodeKind` is used by
/// [`emit_commands`](super::analysis::emit_commands) to produce the final command stream.
#[derive(Debug, Clone)]
#[allow(private_interfaces)]
pub enum NodeKind {
    /// Execute a compute shader.
    Dispatch {
        pipeline: ComputePipelineHandle,
        resource_slots: Vec<u32>,
        user_slots: Vec<u32>,
        dispatch: DispatchDim,
    },
    /// Zero-fill a buffer region (GPU-side clear).
    ClearBuffer {
        buffer: BufferHandle,
        offset: u64,
        size: u64,
    },
    /// Upload CPU data into a buffer, batched with the compute submission.
    WriteBuffer {
        buffer: BufferHandle,
        offset: u64,
        data: Arc<[u8]>,
    },
    /// GPU buffer-to-buffer copy addressed by parcel identity (no payload bytes in the IR).
    CopyBuffer {
        src: ResourceId,
        src_offset: u64,
        dst: ResourceId,
        dst_offset: u64,
        size: u64,
    },
    /// Copy pixel bytes from a CPU-writable buffer into a texture subregion.
    ///
    /// When `src_row_pitch == 0` the source is tightly packed (`width * height * bpp`)
    /// and the backend will repack into a footprint-aligned staging buffer at submit time.
    ///
    /// When `src_row_pitch > 0` the source buffer was already allocated and written with
    /// D3D12 footprint row pitch (and padded to `staging_bytes`), so the backend can skip
    /// the intermediate repack and copy directly from the source parcel.
    CopyBufferToTexture {
        src: ResourceId,
        src_offset: u64,
        /// 0 = tight layout (backend must repack); >0 = footprint row pitch already applied.
        src_row_pitch: u32,
        dst: TextureHandle,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    /// Upload CPU pixel data into a texture (full image), batched with the same GPU submission.
    WriteTexture {
        texture: TextureHandle,
        data: Arc<[u8]>,
        width: u32,
        height: u32,
    },
    /// Upload a subrectangle of a texture.
    WriteTextureRegion {
        texture: TextureHandle,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: Arc<[u8]>,
    },
    /// Copy the full contents of one texture into another.
    ///
    /// Both textures must have compatible formats and identical dimensions.
    /// `src` must have [`crate::types::TextureFlags::COPY_SRC`] and
    /// `dst` must have [`crate::types::TextureFlags::COPY_DST`].
    CopyTexture {
        src: TextureHandle,
        dst: ResourceId,
        /// `Some` when `dst` is a buffer parcel; lowers to image→buffer transfer with this footprint.
        ///
        /// TODO: This duplicates data derivable from the source texture descriptor (`width` /
        /// `height` / `format` on the parcel) plus a backend footprint query at record time
        /// (tight rows on Vulkan/Metal; `GetCopyableFootprints` on DX12). It is stored on the
        /// node so callers can depad CPU reads without a second query and so [`super::analysis`]
        /// can lower without holding a backend handle. Remove by re-querying in the backend when
        /// recording [`crate::backend::GpuCommand::CopyTextureToReadback`] and returning the
        /// footprint only from scheme copy helpers (for client-side depadding after `wait()`).
        dst_buffer_layout: Option<crate::backend::TextureCopyFootprint>,
    },
    /// Copy an offscreen [`crate::RenderTarget`] color buffer to a texture or swapchain output.
    ///
    /// The source render target must have been written by an earlier
    /// [`NodeKind::RenderPass`] node in the same graph. Declare a `Read` binding
    /// on the same [`super::ResourceId::RenderTarget`] so the analyzer orders the
    /// copy after the render pass.
    CopyRenderTarget { src: RenderTargetHandle, dst: ResourceId },
    /// Offscreen render pass targeting a [`crate::RenderTarget`].
    ///
    /// Declare all buffers and textures read by draw commands via
    /// [`super::graph::RenderPassBuilder`] so barriers serialize correctly
    /// against compute work.
    RenderPass {
        target: RenderTargetHandle,
        commands: Vec<crate::backend::RenderCommand>,
    },
    /// Read easement grant — recorded once, replayed with the scheme.
    ///
    /// Emits no GPU commands in v1; exists so the analyzer can eventually
    /// choose host-visible backing vs an inserted device→host blit per backend.
    GrantRead { grant_id: u32 },
    /// Present easement grant — recorded once, replayed with the scheme.
    ///
    /// Emits no GPU commands; owns the ordering edge before scanout return.
    GrantPresent { grant_id: u32 },
}

/// A single node in the task graph.
#[derive(Debug, Clone)]
pub struct TaskNode {
    pub label: &'static str,
    /// Resource access declarations used by the dependency analyzer.
    pub bindings: Vec<ResourceBinding>,
    /// What this node actually executes.
    pub kind: NodeKind,
}

/// The full graph before scheduling.
#[derive(Debug, Clone, Default)]
pub struct GraphIR {
    pub nodes: Vec<TaskNode>,
}

impl GraphIR {
    /// Insert a zero-fill node at the front of the IR so that it executes
    /// before every other node that touches the same parent buffer.
    ///
    /// Used by the graph-colored transient path to guarantee that the
    /// placement-heap region is zeroed before any dispatch reads from it.
    pub fn prepend_clear_buffer(&mut self, buffer: &crate::Buffer, offset: u64, size: u64) {
        let handle = buffer.backing_handle();
        self.nodes.insert(
            0,
            TaskNode {
                label: "clear_transient_region",
                bindings: vec![ResourceBinding {
                    resource: super::ResourceId::Buffer(handle),
                    access: NodeAccess::Write,
                }],
                kind: NodeKind::ClearBuffer {
                    buffer: handle,
                    offset,
                    size,
                },
            },
        );
    }
}

/// Per-resource sync/access semantics on one side of a barrier.
///
/// `src` describes what the producer wave did; `dst` what the consumer wave will do.
/// Backends lower these to native synchronization primitives (DX12 enhanced-barrier
/// sync/access flags, Vulkan stage/access masks).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BarrierUsage {
    /// What the producer side did to this resource.
    pub src: SlotUsageSet,
    /// What the consumer side will do to this resource.
    pub dst: SlotUsageSet,
}

/// Resources that need a barrier before a wave executes, with per-resource access semantics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BarrierSet {
    pub buffers: Vec<(BufferHandle, BarrierUsage)>,
    pub textures: Vec<(TextureHandle, BarrierUsage)>,
    /// Transient buffer IDs whose concrete `BufferHandle` is only known at
    /// emission time (after slot resolution).  Resolved and folded into the
    /// `ResourceBarrier` command inside `emit_waves_to_commands`.
    pub transient_ids: Vec<(u32, BarrierUsage)>,
}

impl BarrierSet {
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty() && self.textures.is_empty() && self.transient_ids.is_empty()
    }
}

/// A group of independent task nodes that can execute concurrently.
#[derive(Debug, Clone)]
pub struct Wave {
    pub node_indices: Vec<usize>,
    pub barriers_before: BarrierSet,
}

/// The result of graph analysis: an ordered sequence of waves with barriers.
#[derive(Debug, Clone)]
pub struct CompiledSchedule {
    pub waves: Vec<Wave>,
}
