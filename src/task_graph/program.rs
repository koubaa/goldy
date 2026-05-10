//! Precompiled compute graphs with abstract buffer/texture slots ([`ComputeProgram`]).
//!
//! Build a graph once using [`ProgramBuilder`] with slot placeholders, then call
//! [`ComputeProgram::specialize`] each frame with concrete handles — wave scheduling
//! and barrier placement are reused without re-running the analyzer.

use super::analysis;
use super::ir::{
    CompiledSchedule, DispatchDim, GraphIR, NodeAccess, NodeKind, ResourceBinding, BarrierSet, TaskNode,
    Wave,
};
use super::ResourceId;
use crate::backend::{BufferHandle, ComputePipelineHandle, TextureHandle};
use crate::buffer::{Buffer, BufferView};
use crate::compute::ComputePipeline;
use crate::texture::Texture;
use anyhow::Result;
use std::collections::HashMap;

/// Slot → concrete resource handles for [`ComputeProgram::specialize`].
#[derive(Debug, Clone, Default)]
pub struct ProgramResolution {
    pub buffers: HashMap<u32, BufferHandle>,
    pub textures: HashMap<u32, TextureHandle>,
}

impl ProgramResolution {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bind_buffer(&mut self, slot: u32, buf: &Buffer) -> &mut Self {
        self.buffers.insert(slot, buf.gpu_buffer_handle());
        self
    }

    pub fn bind_texture(&mut self, slot: u32, tex: &Texture) -> &mut Self {
        self.textures.insert(slot, tex.handle());
        self
    }

    fn buffer(&self, slot: u32) -> Result<BufferHandle> {
        self.buffers
            .get(&slot)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("ProgramResolution: missing buffer slot {}", slot))
    }

    fn texture(&self, slot: u32) -> Result<TextureHandle> {
        self.textures
            .get(&slot)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("ProgramResolution: missing texture slot {}", slot))
    }
}

/// Builder for a [`ComputeProgram`]: static topology with slot placeholders.
pub struct ProgramBuilder {
    ir: GraphIR,
    next_slot: u32,
}

impl ProgramBuilder {
    pub fn new() -> Self {
        Self {
            ir: GraphIR::default(),
            next_slot: 0,
        }
    }

    /// Allocate a buffer slot id for use with [`ProgramStepBuilder::bind_buffer_slot`].
    pub fn buffer_slot(&mut self) -> u32 {
        let s = self.next_slot;
        self.next_slot += 1;
        s
    }

    /// Allocate a texture slot id for use with [`ProgramStepBuilder::bind_texture_slot`].
    pub fn texture_slot(&mut self) -> u32 {
        let s = self.next_slot;
        self.next_slot += 1;
        s
    }

    /// Start a compute step (dispatch node).
    pub fn step<'a>(&'a mut self, label: &str, pipeline: &ComputePipeline) -> ProgramStepBuilder<'a> {
        ProgramStepBuilder {
            builder: self,
            label: label.to_string(),
            pipeline: pipeline.handle,
            bindings: Vec::new(),
            resource_slots: Vec::new(),
            user_slots: Vec::new(),
        }
    }

    /// Analyze the graph once and produce a [`ComputeProgram`].
    pub fn compile(self) -> Result<ComputeProgram> {
        for n in &self.ir.nodes {
            if matches!(n.kind, NodeKind::RenderPass { .. }) {
                anyhow::bail!("ComputeProgram does not support render_pass nodes");
            }
            if n.bindings.iter().any(|b| {
                matches!(
                    b.resource,
                    ResourceId::TransientBuffer(_) | ResourceId::TransientTexture(_)
                )
            }) {
                anyhow::bail!(
                    "ComputeProgram does not support transient_buffer/transient_texture resources"
                );
            }
        }
        let edges = analysis::build_edges(&self.ir);
        let schedule = analysis::schedule_waves(&self.ir, &edges);
        Ok(ComputeProgram {
            ir: self.ir,
            schedule,
        })
    }
}

impl Default for ProgramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-node builder for [`ProgramBuilder::step`].
pub struct ProgramStepBuilder<'a> {
    builder: &'a mut ProgramBuilder,
    label: String,
    pipeline: ComputePipelineHandle,
    bindings: Vec<ResourceBinding>,
    resource_slots: Vec<u32>,
    user_slots: Vec<u32>,
}

impl ProgramStepBuilder<'_> {
    pub fn bind_buffer_slot(mut self, slot: u32, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::ProgramBuffer(slot),
            access,
        });
        self
    }

    pub fn bind_buffer_range_slot(mut self, slot: u32, offset: u64, len: u64, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::ProgramBufferRange { slot, offset, len },
            access,
        });
        self
    }

    pub fn bind_texture_slot(mut self, slot: u32, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::ProgramTexture(slot),
            access,
        });
        self
    }

    /// Declare that this step uses a concrete buffer (fixed for all specializations).
    pub fn bind_buffer(mut self, buf: &Buffer, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::Buffer(buf.handle),
            access,
        });
        self
    }

    pub fn bind_buffer_view(mut self, view: &BufferView, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::BufferRange {
                parent: view.parent_handle(),
                offset: view.offset(),
                len: view.size(),
            },
            access,
        });
        self
    }

    pub fn bind_texture(mut self, tex: &Texture, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::Texture(tex.handle()),
            access,
        });
        self
    }

    pub fn bind_resources_raw(mut self, indices: &[u32]) -> Self {
        self.resource_slots = indices.to_vec();
        self
    }

    pub fn bind_resources_raw_with_user(mut self, indices: &[u32], user: &[u32]) -> Self {
        self.resource_slots = indices.to_vec();
        self.user_slots = user.to_vec();
        self
    }

    pub fn dispatch(self, x: u32, y: u32, z: u32) {
        let node = TaskNode {
            label: self.label,
            bindings: self.bindings,
            kind: NodeKind::Dispatch {
                pipeline: self.pipeline,
                resource_slots: self.resource_slots,
                user_slots: self.user_slots,
                dispatch: DispatchDim::Direct { x, y, z },
            },
        };
        self.builder.ir.nodes.push(node);
    }

    pub fn dispatch_indirect(self, buf: &Buffer, offset: u64) {
        let node = TaskNode {
            label: self.label,
            bindings: self.bindings,
            kind: NodeKind::Dispatch {
                pipeline: self.pipeline,
                resource_slots: self.resource_slots,
                user_slots: self.user_slots,
                dispatch: DispatchDim::Indirect {
                    buffer: buf.handle,
                    offset,
                },
            },
        };
        self.builder.ir.nodes.push(node);
    }
}

/// Cached scheduling for a slot-based compute graph.
pub struct ComputeProgram {
    ir: GraphIR,
    schedule: CompiledSchedule,
}

impl ComputeProgram {
    /// Resolve slots and emit `GpuCommand`s without re-analyzing the graph.
    pub fn specialize(&self, res: &ProgramResolution) -> Result<Vec<crate::backend::GpuCommand>> {
        let ir = substitute_ir(&self.ir, res)?;
        let schedule = resolve_full_schedule(&self.schedule, res)?;
        Ok(analysis::emit_commands(&ir, &schedule))
    }
}

fn resolve_barrier_set(bs: &BarrierSet, res: &ProgramResolution) -> Result<BarrierSet> {
    let mut buffers: Vec<BufferHandle> = bs.buffers.clone();
    for &s in &bs.program_buffer_slots {
        buffers.push(res.buffer(s)?);
    }
    buffers.sort_unstable();
    buffers.dedup();

    let mut textures: Vec<TextureHandle> = bs.textures.clone();
    for &s in &bs.program_texture_slots {
        textures.push(res.texture(s)?);
    }
    textures.sort_unstable();
    textures.dedup();

    Ok(BarrierSet {
        buffers,
        textures,
        program_buffer_slots: Vec::new(),
        program_texture_slots: Vec::new(),
    })
}

fn resolve_full_schedule(
    schedule: &CompiledSchedule,
    res: &ProgramResolution,
) -> Result<CompiledSchedule> {
    let mut waves = Vec::with_capacity(schedule.waves.len());
    for w in &schedule.waves {
        waves.push(Wave {
            node_indices: w.node_indices.clone(),
            barriers_before: resolve_barrier_set(&w.barriers_before, res)?,
        });
    }
    Ok(CompiledSchedule { waves })
}

fn substitute_rid(r: ResourceId, res: &ProgramResolution) -> Result<ResourceId> {
    Ok(match r {
        ResourceId::ProgramBuffer(s) => ResourceId::Buffer(res.buffer(s)?),
        ResourceId::ProgramTexture(s) => ResourceId::Texture(res.texture(s)?),
        ResourceId::ProgramBufferRange { slot, offset, len } => ResourceId::BufferRange {
            parent: res.buffer(slot)?,
            offset,
            len,
        },
        o => o,
    })
}

fn substitute_ir(ir: &GraphIR, res: &ProgramResolution) -> Result<GraphIR> {
    let mut nodes = Vec::with_capacity(ir.nodes.len());
    for n in &ir.nodes {
        let bindings: Result<Vec<_>> = n
            .bindings
            .iter()
            .map(|b| {
                Ok(ResourceBinding {
                    resource: substitute_rid(b.resource, res)?,
                    access: b.access,
                })
            })
            .collect();
        let bindings = bindings?;
        let kind = substitute_kind(&n.kind, res)?;
        nodes.push(TaskNode {
            label: n.label.clone(),
            bindings,
            kind,
        });
    }
    Ok(GraphIR { nodes })
}

fn substitute_kind(kind: &NodeKind, _res: &ProgramResolution) -> Result<NodeKind> {
    Ok(match kind {
        NodeKind::Dispatch {
            pipeline,
            resource_slots,
            user_slots,
            dispatch,
        } => NodeKind::Dispatch {
            pipeline: *pipeline,
            resource_slots: resource_slots.clone(),
            user_slots: user_slots.clone(),
            dispatch: match dispatch {
                DispatchDim::Direct { x, y, z } => DispatchDim::Direct {
                    x: *x,
                    y: *y,
                    z: *z,
                },
                DispatchDim::Indirect { buffer, offset } => DispatchDim::Indirect {
                    buffer: *buffer,
                    offset: *offset,
                },
            },
        },
        NodeKind::ClearBuffer {
            buffer,
            offset,
            size,
        } => NodeKind::ClearBuffer {
            buffer: *buffer,
            offset: *offset,
            size: *size,
        },
        NodeKind::WriteBuffer {
            buffer,
            offset,
            data,
        } => NodeKind::WriteBuffer {
            buffer: *buffer,
            offset: *offset,
            data: data.clone(),
        },
        NodeKind::WriteTexture {
            texture,
            data,
            width,
            height,
        } => NodeKind::WriteTexture {
            texture: *texture,
            data: data.clone(),
            width: *width,
            height: *height,
        },
        NodeKind::WriteTextureRegion {
            texture,
            x,
            y,
            width,
            height,
            data,
        } => NodeKind::WriteTextureRegion {
            texture: *texture,
            x: *x,
            y: *y,
            width: *width,
            height: *height,
            data: data.clone(),
        },
        NodeKind::RenderPass { .. } => anyhow::bail!("RenderPass in substitute_kind"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::backend::GpuCommand;
    use crate::compute::ComputePipeline;
    use crate::device::Device;
    use crate::shader::ShaderModule;

    fn device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    #[test]
    fn specialize_twice_same_length() {
        let dev = device();
        let shader = ShaderModule::from_slang(&dev, "void main() {}").unwrap();
        let pipe = ComputePipeline::new(&dev, &shader).unwrap();
        let mut builder = ProgramBuilder::new();
        let s0 = builder.buffer_slot();
        builder
            .step("w", &pipe)
            .bind_buffer_slot(s0, NodeAccess::Write)
            .bind_resources_raw(&[0])
            .dispatch(1, 1, 1);
        builder
            .step("r", &pipe)
            .bind_buffer_slot(s0, NodeAccess::Read)
            .bind_resources_raw(&[0])
            .dispatch(1, 1, 1);
        let prog = builder.compile().unwrap();
        let buf = crate::buffer::Buffer::new(&dev, 256, crate::DataAccess::Scattered).unwrap();
        let mut res = ProgramResolution::new();
        res.bind_buffer(s0, &buf);
        let a = prog.specialize(&res).unwrap();
        let b = prog.specialize(&res).unwrap();
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter()
                .any(|c| matches!(c, GpuCommand::ResourceBarrier { .. })),
            "expected a barrier between write and read waves"
        );
    }
}
