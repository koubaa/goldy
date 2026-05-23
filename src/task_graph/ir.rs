//! Task graph intermediate representation: nodes, bindings, and compiled schedules.
//!
//! These types form the shared IR consumed by [`analysis`](super::analysis).
//! The [`analysis`](super::analysis) module consumes a [`GraphIR`] and produces a
//! [`CompiledSchedule`] of [`Wave`]s with [`BarrierSet`]s.
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
    /// Flags are unioned across all edges contributing to a given [`BarrierSet`].
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
/// (dst) to the slots covered by a [`BarrierSet`].  Each backend lowers this to
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
/// resource's physical [`DataAccess`](crate::DataAccess) /
/// [`SpatialAccess`](crate::SpatialAccess).
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
        dst: TextureHandle,
    },
    /// Offscreen render pass targeting a [`crate::RenderTarget`].
    ///
    /// Declare all buffers and textures read by draw commands via
    /// [`super::graph::RenderPassBuilder`] so barriers serialize correctly
    /// against compute work.
    RenderPass {
        target: RenderTargetHandle,
        commands: Vec<crate::backend::RenderCommand>,
    },
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
        self.nodes.insert(
            0,
            TaskNode {
                label: "clear_transient_region",
                bindings: vec![ResourceBinding {
                    resource: super::ResourceId::Buffer(buffer.handle),
                    access: NodeAccess::Write,
                }],
                kind: NodeKind::ClearBuffer {
                    buffer: buffer.handle,
                    offset,
                    size,
                },
            },
        );
    }
}

/// Resources that need a barrier before a wave executes, with full access semantics.
///
/// `src_usage` and `dst_usage` describe *what kind of GPU work* produced and will
/// consume the listed resources.  Backends lower these to native synchronization
/// primitives (DX12 enhanced-barrier sync/access flags, Vulkan stage/access masks).
///
/// Both fields are the *union* across all dependency edges that feed this barrier:
/// if wave N depends on both a `ClearBuffer` (Transfer/Write) and a `Dispatch`
/// (Compute/Write), then `src_usage.kinds` contains `TRANSFER | COMPUTE`.
#[derive(Debug, Clone, Default)]
pub struct BarrierSet {
    pub buffers: Vec<BufferHandle>,
    pub textures: Vec<TextureHandle>,
    /// Transient buffer IDs whose concrete `BufferHandle` is only known at
    /// emission time (after slot resolution).  Resolved and folded into the
    /// `ResourceBarrier` command inside `emit_waves_to_commands`.
    pub transient_ids: Vec<u32>,
    /// What the producer side did to the listed resources.
    pub src_usage: SlotUsageSet,
    /// What the consumer side will do to the listed resources.
    pub dst_usage: SlotUsageSet,
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
