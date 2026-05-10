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
use crate::backend::{BufferHandle, ComputePipelineHandle, TextureHandle};
use std::sync::Arc;

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
#[derive(Debug, Clone)]
pub struct ResourceBinding {
    pub resource: ResourceId,
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
}

/// A single node in the task graph.
#[derive(Debug, Clone)]
pub struct TaskNode {
    #[allow(dead_code)]
    pub label: String,
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

/// Resources that need a barrier before a wave executes.
#[derive(Debug, Clone, Default)]
pub struct BarrierSet {
    pub buffers: Vec<BufferHandle>,
    pub textures: Vec<TextureHandle>,
}

impl BarrierSet {
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty() && self.textures.is_empty()
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
