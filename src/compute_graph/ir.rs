//! Graph intermediate representation: nodes, bindings, and compiled schedules.
//!
//! These types form the shared IR used by both [`ComputeGraph`](super::ComputeGraph)
//! (Tier 1) and [`ComputeProgram`](super::ComputeProgram) (Tier 2). The
//! [`analysis`](super::analysis) module consumes a [`GraphIR`] and produces a
//! [`CompiledSchedule`] of [`Wave`]s with [`BarrierSet`]s.

use super::ResourceId;
use crate::backend::{BufferHandle, ComputePipelineHandle, TextureHandle};

/// Logical access a dispatch node has on a resource, orthogonal to the
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

/// A single resource binding within a graph node.
#[derive(Debug, Clone)]
pub struct ResourceBinding {
    pub resource: ResourceId,
    pub access: NodeAccess,
}

/// How a graph node's dispatch dimensions are specified.
#[derive(Debug, Clone)]
pub enum DispatchKind {
    /// Fixed workgroup counts known at graph construction time.
    Direct { x: u32, y: u32, z: u32 },
    /// Workgroup counts read from a buffer at runtime (3× `u32` at `offset`).
    Indirect { buffer: BufferHandle, offset: u64 },
}

/// A dispatch node in the graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    #[allow(dead_code)]
    pub label: String,
    pub pipeline: ComputePipelineHandle,
    pub bindings: Vec<ResourceBinding>,
    pub push_constants: Vec<u32>,
    pub dispatch: DispatchKind,
}

/// The full graph before scheduling.
#[derive(Debug, Clone, Default)]
pub struct GraphIR {
    pub nodes: Vec<GraphNode>,
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

/// A group of independent dispatch nodes that can execute concurrently.
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
