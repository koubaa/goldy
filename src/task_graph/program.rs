//! Precompiled task graphs with abstract buffer/texture/render-target slots ([`GraphProgram`]).
//!
//! Build a graph once using [`ProgramBuilder`] with slot placeholders, then call
//! [`GraphProgram::specialize`] (or [`GraphProgram::specialize_graph`] for mixed
//! compute + render programs) each frame with concrete handles — wave scheduling
//! and barrier placement are reused without re-running the analyzer.

use super::analysis;
use super::ir::{
    BarrierSet, CompiledSchedule, DispatchDim, GraphIR, NodeAccess, NodeKind, ResourceBinding,
    TaskNode, Wave,
};
use super::ResourceId;
use crate::backend::{BufferHandle, ComputePipelineHandle, RenderTargetHandle, TextureHandle};
use crate::buffer::{Buffer, BufferView};
use crate::compute::ComputePipeline;
use crate::render_target::RenderTarget;
use crate::texture::Texture;
use anyhow::Result;
use std::collections::HashMap;

/// Slot → concrete resource handles for [`GraphProgram::specialize`].
#[derive(Debug, Clone, Default)]
pub struct ProgramResolution {
    pub buffers: HashMap<u32, BufferHandle>,
    pub textures: HashMap<u32, TextureHandle>,
    pub render_targets: HashMap<u32, RenderTargetHandle>,
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

    pub fn bind_render_target(&mut self, slot: u32, target: &RenderTarget) -> &mut Self {
        self.render_targets.insert(slot, target.backend_handle());
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

    fn render_target(&self, slot: u32) -> Result<RenderTargetHandle> {
        self.render_targets.get(&slot).copied().ok_or_else(|| {
            anyhow::anyhow!("ProgramResolution: missing render target slot {}", slot)
        })
    }
}

/// Builder for a [`GraphProgram`]: static topology with slot placeholders.
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

    /// Allocate a render target slot id for use with [`ProgramBuilder::render_pass_step`].
    pub fn render_target_slot(&mut self) -> u32 {
        let s = self.next_slot;
        self.next_slot += 1;
        s
    }

    /// Start a compute step (dispatch node).
    pub fn step<'a>(
        &'a mut self,
        label: &str,
        pipeline: &ComputePipeline,
    ) -> ProgramStepBuilder<'a> {
        ProgramStepBuilder {
            builder: self,
            label: label.to_string(),
            pipeline: pipeline.handle,
            bindings: Vec::new(),
            resource_slots: Vec::new(),
            user_slots: Vec::new(),
        }
    }

    /// Start a render pass step targeting a render-target slot.
    ///
    /// The `target_slot` should be allocated via [`Self::render_target_slot`]
    /// and resolved at specialization time via
    /// [`ProgramResolution::bind_render_target`].
    pub fn render_pass_step<'a>(
        &'a mut self,
        label: &str,
        target_slot: u32,
    ) -> ProgramRenderPassBuilder<'a> {
        ProgramRenderPassBuilder {
            builder: self,
            label: label.to_string(),
            target_slot,
            bindings: Vec::new(),
        }
    }

    /// Analyze the graph once and produce a [`GraphProgram`].
    pub fn compile(self) -> Result<GraphProgram> {
        let mut has_render = false;
        for n in &self.ir.nodes {
            if matches!(n.kind, NodeKind::RenderPass { .. }) {
                has_render = true;
            }
            if n.bindings.iter().any(|b| {
                matches!(
                    b.resource,
                    ResourceId::TransientBuffer(_) | ResourceId::TransientTexture(_)
                )
            }) {
                anyhow::bail!(
                    "GraphProgram does not support transient_buffer/transient_texture resources"
                );
            }
        }
        let edges = analysis::build_edges(&self.ir);
        let schedule = analysis::schedule_waves(&self.ir, &edges);
        Ok(GraphProgram {
            ir: self.ir,
            schedule,
            has_render,
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

    pub fn bind_buffer_range_slot(
        mut self,
        slot: u32,
        offset: u64,
        len: u64,
        access: NodeAccess,
    ) -> Self {
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

/// Per-node builder for [`ProgramBuilder::render_pass_step`].
pub struct ProgramRenderPassBuilder<'a> {
    builder: &'a mut ProgramBuilder,
    label: String,
    target_slot: u32,
    bindings: Vec<ResourceBinding>,
}

impl ProgramRenderPassBuilder<'_> {
    pub fn bind_buffer_slot(mut self, slot: u32, access: NodeAccess) -> Self {
        self.bindings.push(ResourceBinding {
            resource: ResourceId::ProgramBuffer(slot),
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

    /// Finalize the render pass node with fixed [`RenderCommand`](crate::backend::RenderCommand)s.
    ///
    /// The render target will be resolved from the slot at specialization time;
    /// the draw commands are baked into the program and reused across specializations.
    pub fn finish(self, commands: Vec<crate::backend::RenderCommand>) {
        // Store the target_slot in the RenderTargetHandle field as a sentinel;
        // substitute_kind resolves it to the concrete handle at specialize time.
        self.builder.ir.nodes.push(TaskNode {
            label: self.label,
            bindings: self.bindings,
            kind: NodeKind::RenderPass {
                target: self.target_slot as RenderTargetHandle,
                commands,
            },
        });
    }
}

/// Cached scheduling for a slot-based graph (compute and/or render pass nodes).
///
/// Build with [`ProgramBuilder`], compile once, then call [`specialize`](Self::specialize)
/// each frame to resolve slot placeholders without re-running the analyzer.
pub struct GraphProgram {
    ir: GraphIR,
    schedule: CompiledSchedule,
    has_render: bool,
}

impl GraphProgram {
    /// Resolve slots and emit `GpuCommand`s without re-analyzing the graph.
    ///
    /// Panics if the program contains render pass nodes — use
    /// [`specialize_graph`](Self::specialize_graph) for mixed programs.
    pub fn specialize(&self, res: &ProgramResolution) -> Result<Vec<crate::backend::GpuCommand>> {
        assert!(
            !self.has_render,
            "specialize: program contains render passes; use specialize_graph"
        );
        let ir = substitute_ir(&self.ir, res)?;
        let schedule = resolve_full_schedule(&self.schedule, res)?;
        Ok(analysis::emit_commands(&ir, &schedule))
    }

    /// Resolve slots and emit [`GraphCommand`](crate::backend::GraphCommand)s for mixed programs.
    pub fn specialize_graph(
        &self,
        res: &ProgramResolution,
    ) -> Result<Vec<crate::backend::GraphCommand>> {
        let ir = substitute_ir(&self.ir, res)?;
        let schedule = resolve_full_schedule(&self.schedule, res)?;
        Ok(analysis::emit_graph_commands(&ir, &schedule))
    }

    /// Whether this program contains any render pass nodes.
    pub fn has_render_passes(&self) -> bool {
        self.has_render
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
        ResourceId::RenderTargetSlot(_) => r,
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

fn substitute_kind(kind: &NodeKind, res: &ProgramResolution) -> Result<NodeKind> {
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
        NodeKind::RenderPass { target, commands } => {
            let slot = *target as u32;
            let resolved_target = res.render_target(slot)?;
            NodeKind::RenderPass {
                target: resolved_target,
                commands: commands.clone(),
            }
        }
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

    #[test]
    fn mixed_compute_render_graph_program() {
        use crate::backend::{GraphCommand, RenderCommand};
        use crate::render_target::RenderTarget;
        use crate::types::{Color, TextureFormat};

        let dev = device();
        let shader = ShaderModule::from_slang(&dev, "void main() {}").unwrap();
        let pipe = ComputePipeline::new(&dev, &shader).unwrap();

        let mut builder = ProgramBuilder::new();
        let buf_slot = builder.buffer_slot();
        let rt_slot = builder.render_target_slot();

        // Compute step writes to buffer.
        builder
            .step("compute_write", &pipe)
            .bind_buffer_slot(buf_slot, NodeAccess::Write)
            .bind_resources_raw(&[0])
            .dispatch(1, 1, 1);

        // Render pass reads from buffer.
        builder
            .render_pass_step("draw", rt_slot)
            .bind_buffer_slot(buf_slot, NodeAccess::Read)
            .finish(vec![RenderCommand::Clear(Color::RED)]);

        let prog = builder.compile().unwrap();
        assert!(prog.has_render_passes());

        // Specialize with two different render targets.
        let rt_a = RenderTarget::new(&dev, 8, 8, TextureFormat::Rgba8Unorm).unwrap();
        let rt_b = RenderTarget::new(&dev, 16, 16, TextureFormat::Rgba8Unorm).unwrap();
        let buf = crate::buffer::Buffer::new(&dev, 256, crate::DataAccess::Scattered).unwrap();

        let mut res_a = ProgramResolution::new();
        res_a.bind_buffer(buf_slot, &buf);
        res_a.bind_render_target(rt_slot, &rt_a);

        let mut res_b = ProgramResolution::new();
        res_b.bind_buffer(buf_slot, &buf);
        res_b.bind_render_target(rt_slot, &rt_b);

        let cmds_a = prog.specialize_graph(&res_a).unwrap();
        let cmds_b = prog.specialize_graph(&res_b).unwrap();

        // Both should have the same structure (same number of commands).
        assert_eq!(cmds_a.len(), cmds_b.len());

        // Both should contain compute and render commands.
        assert!(
            cmds_a.iter().any(|c| matches!(c, GraphCommand::Compute(_))),
            "expected Compute commands"
        );
        assert!(
            cmds_a
                .iter()
                .any(|c| matches!(c, GraphCommand::Render { .. })),
            "expected Render commands"
        );

        // Should have a barrier between compute write and render read.
        assert!(
            cmds_a
                .iter()
                .any(|c| matches!(c, GraphCommand::Compute(GpuCommand::ResourceBarrier { .. }))),
            "expected ResourceBarrier between compute and render waves"
        );

        // The two specializations should target different render targets.
        let target_a = cmds_a
            .iter()
            .find_map(|c| match c {
                GraphCommand::Render { target, .. } => Some(*target),
                _ => None,
            })
            .unwrap();
        let target_b = cmds_b
            .iter()
            .find_map(|c| match c {
                GraphCommand::Render { target, .. } => Some(*target),
                _ => None,
            })
            .unwrap();
        assert_ne!(
            target_a, target_b,
            "specializations should resolve to different render targets"
        );
    }
}
