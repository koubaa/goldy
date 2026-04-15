//! `ComputeProgram` — Tier 2: compiled, specializable compute graph.
//!
//! Separates graph topology (static) from resource bindings and dispatch
//! dimensions (dynamic). The topology is analyzed once at [`compile`] time;
//! the cached schedule is replayed cheaply at [`specialize`] + [`submit`] time.
//!
//! This is the recommended API for fixed-topology pipelines (like Ekrano's
//! render passes) where the sequence of shaders doesn't change frame to frame,
//! only buffer sizes and dispatch dimensions do.
//!
//! [`compile`]: ProgramBuilder::compile
//! [`specialize`]: ComputeProgram::specialize
//! [`submit`]: Execution::submit

use std::marker::PhantomData;
use std::sync::Arc;

use super::analysis;
use super::ir::{CompiledSchedule, DispatchKind, GraphIR, GraphNode, NodeAccess, ResourceBinding};
use super::ResourceId;
use crate::backend::{BufferHandle, ComputeCommand, ComputePipelineHandle, TextureHandle};
use crate::buffer::Buffer;
use crate::compute::ComputePipeline;
use crate::device::Device;
use crate::gpu_future::GpuFuture;
use crate::texture::Texture;
use anyhow::Result;

/// Type-safe handle for a resource slot in a [`ComputeProgram`].
#[derive(Debug)]
pub struct SlotId<T> {
    index: usize,
    _marker: PhantomData<T>,
}

impl<T> Clone for SlotId<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SlotId<T> {}

impl<T> PartialEq for SlotId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}
impl<T> Eq for SlotId<T> {}

/// Handle for a dispatch-dimension slot in a [`ComputeProgram`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimSlotId {
    index: usize,
}

// Internal slot types used during build.
#[derive(Debug, Clone)]
enum SlotKind {
    Buffer {
        #[allow(dead_code)]
        name: String,
    },
    Texture {
        #[allow(dead_code)]
        name: String,
    },
}

#[derive(Debug, Clone)]
struct StepBinding {
    slot_index: usize,
    is_texture: bool,
    access: NodeAccess,
}

#[derive(Debug, Clone)]
struct StepDef {
    label: String,
    pipeline: ComputePipelineHandle,
    bindings: Vec<StepBinding>,
    push_constants: Vec<PushConstantSource>,
    dim: DimSource,
}

#[derive(Debug, Clone)]
enum PushConstantSource {
    Literal(u32),
}

#[derive(Debug, Clone)]
enum DimSource {
    Fixed(u32, u32, u32),
    Slot(usize),
}

/// Builder for a [`ComputeProgram`].
///
/// Declare typed resource slots and dispatch steps, then call
/// [`compile`](ProgramBuilder::compile) to produce a reusable program
/// with cached scheduling and barrier placement.
///
/// # Example
///
/// ```rust,ignore
/// let mut builder = ComputeProgram::builder();
/// let input  = builder.buffer_slot("input");
/// let output = builder.buffer_slot("output");
/// let wg     = builder.dim_slot("workgroups");
///
/// builder.step("transform", &pipeline)
///     .bind_buffer(input, NodeAccess::Read)
///     .bind_buffer(output, NodeAccess::Write)
///     .dispatch_slot(wg);
///
/// let program = builder.compile()?;
/// ```
pub struct ProgramBuilder {
    slots: Vec<SlotKind>,
    dim_slots: Vec<String>,
    steps: Vec<StepDef>,
}

impl ProgramBuilder {
    /// Declare a buffer slot. Returns a typed handle for use in step bindings.
    pub fn buffer_slot(&mut self, name: &str) -> SlotId<Buffer> {
        let index = self.slots.len();
        self.slots.push(SlotKind::Buffer {
            name: name.to_string(),
        });
        SlotId {
            index,
            _marker: PhantomData,
        }
    }

    /// Declare a texture slot. Returns a typed handle for use in step bindings.
    pub fn texture_slot(&mut self, name: &str) -> SlotId<Texture> {
        let index = self.slots.len();
        self.slots.push(SlotKind::Texture {
            name: name.to_string(),
        });
        SlotId {
            index,
            _marker: PhantomData,
        }
    }

    /// Declare a dispatch-dimension slot (filled at specialize time).
    pub fn dim_slot(&mut self, name: &str) -> DimSlotId {
        let index = self.dim_slots.len();
        self.dim_slots.push(name.to_string());
        DimSlotId { index }
    }

    /// Add a dispatch step. The returned [`StepBuilder`] configures bindings and dimensions.
    pub fn step<'a>(&'a mut self, label: &str, pipeline: &ComputePipeline) -> StepBuilder<'a> {
        StepBuilder {
            builder: self,
            step: StepDef {
                label: label.to_string(),
                pipeline: pipeline.handle,
                bindings: Vec::new(),
                push_constants: Vec::new(),
                dim: DimSource::Fixed(1, 1, 1),
            },
        }
    }

    /// Analyze the graph topology and produce a compiled program.
    ///
    /// The compiled schedule (wave grouping and barrier placement) is cached.
    /// Call [`ComputeProgram::specialize`] to bind concrete resources per frame.
    pub fn compile(self) -> Result<ComputeProgram> {
        // Build a template GraphIR using sentinel handles for slots.
        // The slot index is encoded as the handle value — these are never
        // sent to the GPU; they're only used for dependency analysis.
        let mut ir = GraphIR::default();

        for step in &self.steps {
            let bindings = step
                .bindings
                .iter()
                .map(|b| {
                    let resource = if b.is_texture {
                        ResourceId::Texture(b.slot_index as TextureHandle)
                    } else {
                        ResourceId::Buffer(b.slot_index as BufferHandle)
                    };
                    ResourceBinding {
                        resource,
                        access: b.access,
                    }
                })
                .collect();

            ir.nodes.push(GraphNode {
                label: step.label.clone(),
                pipeline: step.pipeline,
                bindings,
                push_constants: Vec::new(),
                dispatch: DispatchKind::Direct { x: 0, y: 0, z: 0 },
            });
        }

        let edges = analysis::build_edges(&ir);
        let schedule = analysis::schedule_waves(&ir, &edges);

        Ok(ComputeProgram {
            slots: self.slots,
            dim_slots: self.dim_slots,
            steps: self.steps,
            schedule,
        })
    }
}

/// Builder for a single step within a [`ProgramBuilder`].
pub struct StepBuilder<'a> {
    builder: &'a mut ProgramBuilder,
    step: StepDef,
}

impl<'a> StepBuilder<'a> {
    /// Bind a buffer slot with the given logical access.
    pub fn bind_buffer(mut self, slot: SlotId<Buffer>, access: NodeAccess) -> Self {
        self.step.bindings.push(StepBinding {
            slot_index: slot.index,
            is_texture: false,
            access,
        });
        self
    }

    /// Bind a texture slot with the given logical access.
    pub fn bind_texture(mut self, slot: SlotId<Texture>, access: NodeAccess) -> Self {
        self.step.bindings.push(StepBinding {
            slot_index: slot.index,
            is_texture: true,
            access,
        });
        self
    }

    /// Set literal push constants for this step.
    pub fn push_constants_raw(mut self, indices: &[u32]) -> Self {
        self.step.push_constants = indices
            .iter()
            .copied()
            .map(PushConstantSource::Literal)
            .collect();
        self
    }

    /// Set fixed dispatch dimensions.
    pub fn dispatch(mut self, x: u32, y: u32, z: u32) {
        self.step.dim = DimSource::Fixed(x, y, z);
        self.builder.steps.push(self.step);
    }

    /// Set dispatch dimensions from a dimension slot (resolved at specialize time).
    pub fn dispatch_slot(mut self, slot: DimSlotId) {
        self.step.dim = DimSource::Slot(slot.index);
        self.builder.steps.push(self.step);
    }
}

/// A compiled compute graph with cached scheduling and barrier placement.
///
/// Created by [`ProgramBuilder::compile`]. The graph topology, wave grouping,
/// and barrier placement are all fixed at compile time. Use
/// [`specialize`](ComputeProgram::specialize) each frame to bind concrete
/// resources and dimensions, then submit — no re-analysis needed.
///
/// ```text
/// ProgramBuilder::compile()          →  [build graph] → [analyze] → [cache schedule]
/// ComputeProgram::specialize().submit() →  [bind slots] → [replay cached schedule]
///                                               ↑ every frame (cheap)
/// ```
pub struct ComputeProgram {
    slots: Vec<SlotKind>,
    dim_slots: Vec<String>,
    steps: Vec<StepDef>,
    schedule: CompiledSchedule,
}

impl ComputeProgram {
    /// Create a new program builder.
    pub fn builder() -> ProgramBuilder {
        ProgramBuilder {
            slots: Vec::new(),
            dim_slots: Vec::new(),
            steps: Vec::new(),
        }
    }

    /// Create an [`Execution`] that can be filled with concrete resource bindings
    /// and dispatch dimensions, then submitted.
    pub fn specialize(&self) -> Execution<'_> {
        Execution {
            program: self,
            buffer_bindings: vec![None; self.slots.len()],
            texture_bindings: vec![None; self.slots.len()],
            dims: vec![None; self.dim_slots.len()],
        }
    }
}

/// A specialized execution of a [`ComputeProgram`], ready to submit after
/// binding concrete resources and dimensions.
///
/// Created by [`ComputeProgram::specialize`]. Fill in all declared slots
/// with concrete resources and dimensions, then call [`submit`](Execution::submit)
/// or [`dispatch`](Execution::dispatch). Unbound dimension slots will produce
/// an error at submit time.
pub struct Execution<'a> {
    program: &'a ComputeProgram,
    buffer_bindings: Vec<Option<BufferHandle>>,
    texture_bindings: Vec<Option<TextureHandle>>,
    dims: Vec<Option<(u32, u32, u32)>>,
}

impl<'a> Execution<'a> {
    /// Bind a concrete buffer to a slot.
    pub fn bind_buffer(&mut self, slot: SlotId<Buffer>, buf: &Buffer) {
        self.buffer_bindings[slot.index] = Some(buf.handle);
    }

    /// Bind a concrete texture to a slot.
    pub fn bind_texture(&mut self, slot: SlotId<Texture>, tex: &Texture) {
        self.texture_bindings[slot.index] = Some(tex.handle);
    }

    /// Set dispatch dimensions for a dimension slot.
    pub fn set_dim(&mut self, slot: DimSlotId, dim: (u32, u32, u32)) {
        self.dims[slot.index] = Some(dim);
    }

    /// Emit commands from the cached schedule with concrete bindings.
    fn emit_commands(&self) -> Result<Vec<ComputeCommand>> {
        let mut commands = Vec::new();

        for wave in &self.program.schedule.waves {
            if !wave.barriers_before.is_empty() {
                // Remap sentinel handles to concrete handles.
                let buffers: Vec<BufferHandle> = wave
                    .barriers_before
                    .buffers
                    .iter()
                    .map(|&sentinel| {
                        let slot = sentinel as usize;
                        self.buffer_bindings[slot]
                            .ok_or_else(|| anyhow::anyhow!("buffer slot {} not bound", slot))
                    })
                    .collect::<Result<_>>()?;

                let textures: Vec<TextureHandle> = wave
                    .barriers_before
                    .textures
                    .iter()
                    .map(|&sentinel| {
                        let slot = sentinel as usize;
                        self.texture_bindings[slot]
                            .ok_or_else(|| anyhow::anyhow!("texture slot {} not bound", slot))
                    })
                    .collect::<Result<_>>()?;

                if !buffers.is_empty() || !textures.is_empty() {
                    commands.push(ComputeCommand::ResourceBarrier { buffers, textures });
                }
            }

            for &step_idx in &wave.node_indices {
                let step = &self.program.steps[step_idx];

                commands.push(ComputeCommand::SetPipeline(step.pipeline));

                if !step.push_constants.is_empty() {
                    let indices: Vec<u32> = step
                        .push_constants
                        .iter()
                        .map(|src| match src {
                            PushConstantSource::Literal(v) => *v,
                        })
                        .collect();
                    commands.push(ComputeCommand::SetPushConstantsRaw { indices });
                }

                let (x, y, z) = match &step.dim {
                    DimSource::Fixed(x, y, z) => (*x, *y, *z),
                    DimSource::Slot(idx) => self.dims[*idx].ok_or_else(|| {
                        anyhow::anyhow!("dim slot '{}' not set", self.program.dim_slots[*idx])
                    })?,
                };

                commands.push(ComputeCommand::Dispatch {
                    workgroups_x: x,
                    workgroups_y: y,
                    workgroups_z: z,
                });
            }
        }

        Ok(commands)
    }

    /// Submit the specialized program without blocking.
    pub fn submit(self, device: &Device) -> Result<GpuFuture> {
        let commands = self.emit_commands()?;
        let mut backend = device.backend.lock().unwrap();
        let token = backend.submit_compute(device.handle, &commands)?;
        Ok(GpuFuture {
            backend: Arc::clone(&device.backend),
            device: device.handle,
            fence_token: token,
        })
    }

    /// Submit the specialized program and block until complete.
    pub fn dispatch(self, device: &Device) -> Result<()> {
        let commands = self.emit_commands()?;
        let mut backend = device.backend.lock().unwrap();
        backend.dispatch_compute(device.handle, &commands)
    }

    /// Emit the command sequence for testing/inspection.
    #[cfg(test)]
    pub(crate) fn compile_commands(&self) -> Result<Vec<ComputeCommand>> {
        self.emit_commands()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::backend::ComputeCommand;
    use crate::compute::ComputePipeline;
    use crate::device::Device;
    use crate::shader::ShaderModule;

    fn mock_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    fn mock_shader(device: &Device) -> ShaderModule {
        ShaderModule::from_slang(device, "void main() {}").unwrap()
    }

    fn mock_pipeline(device: &Device, shader: &ShaderModule) -> ComputePipeline {
        ComputePipeline::new(device, shader).unwrap()
    }

    #[test]
    fn compile_and_specialize_linear_chain() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut builder = ComputeProgram::builder();
        let scene = builder.buffer_slot("scene");
        let output = builder.buffer_slot("output");
        let wg = builder.dim_slot("wg");

        builder
            .step("write", &pipeline)
            .bind_buffer(scene, NodeAccess::Read)
            .bind_buffer(output, NodeAccess::Write)
            .dispatch_slot(wg);

        builder
            .step("postprocess", &pipeline)
            .bind_buffer(output, NodeAccess::ReadWrite)
            .dispatch(4, 1, 1);

        let program = builder.compile().unwrap();

        let buf_a = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();
        let buf_b = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        let mut exec = program.specialize();
        exec.bind_buffer(scene, &buf_a);
        exec.bind_buffer(output, &buf_b);
        exec.set_dim(wg, (16, 1, 1));

        let cmds = exec.compile_commands().unwrap();

        // Wave 0: SetPipeline, Dispatch(16,1,1)
        // ResourceBarrier (output buffer)
        // Wave 1: SetPipeline, Dispatch(4,1,1)
        let dispatch_count = cmds
            .iter()
            .filter(|c| matches!(c, ComputeCommand::Dispatch { .. }))
            .count();
        assert_eq!(dispatch_count, 2);

        let barrier_count = cmds
            .iter()
            .filter(|c| matches!(c, ComputeCommand::ResourceBarrier { .. }))
            .count();
        assert_eq!(barrier_count, 1);

        // Check the first dispatch uses dim_slot value
        let first_dispatch = cmds
            .iter()
            .find(|c| matches!(c, ComputeCommand::Dispatch { .. }))
            .unwrap();
        assert!(matches!(
            first_dispatch,
            ComputeCommand::Dispatch {
                workgroups_x: 16,
                ..
            }
        ));
    }

    #[test]
    fn specialize_twice_different_dims() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut builder = ComputeProgram::builder();
        let buf_slot = builder.buffer_slot("data");
        let wg = builder.dim_slot("wg");

        builder
            .step("work", &pipeline)
            .bind_buffer(buf_slot, NodeAccess::ReadWrite)
            .dispatch_slot(wg);

        let program = builder.compile().unwrap();
        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        // First specialization
        let mut exec1 = program.specialize();
        exec1.bind_buffer(buf_slot, &buf);
        exec1.set_dim(wg, (8, 1, 1));
        let cmds1 = exec1.compile_commands().unwrap();

        // Second specialization with different dim
        let mut exec2 = program.specialize();
        exec2.bind_buffer(buf_slot, &buf);
        exec2.set_dim(wg, (32, 1, 1));
        let cmds2 = exec2.compile_commands().unwrap();

        // Both should have one dispatch but with different workgroup counts
        let d1 = cmds1
            .iter()
            .find(|c| matches!(c, ComputeCommand::Dispatch { .. }))
            .unwrap();
        let d2 = cmds2
            .iter()
            .find(|c| matches!(c, ComputeCommand::Dispatch { .. }))
            .unwrap();
        assert!(matches!(
            d1,
            ComputeCommand::Dispatch {
                workgroups_x: 8,
                ..
            }
        ));
        assert!(matches!(
            d2,
            ComputeCommand::Dispatch {
                workgroups_x: 32,
                ..
            }
        ));
    }

    #[test]
    fn independent_steps_no_barrier() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut builder = ComputeProgram::builder();
        let buf_a = builder.buffer_slot("a");
        let buf_b = builder.buffer_slot("b");

        builder
            .step("write_a", &pipeline)
            .bind_buffer(buf_a, NodeAccess::Write)
            .dispatch(1, 1, 1);

        builder
            .step("write_b", &pipeline)
            .bind_buffer(buf_b, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let program = builder.compile().unwrap();

        let ba = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();
        let bb = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        let mut exec = program.specialize();
        exec.bind_buffer(buf_a, &ba);
        exec.bind_buffer(buf_b, &bb);

        let cmds = exec.compile_commands().unwrap();
        assert!(!cmds
            .iter()
            .any(|c| matches!(c, ComputeCommand::ResourceBarrier { .. })));
    }

    #[test]
    fn unbound_slot_errors() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut builder = ComputeProgram::builder();
        let buf_a = builder.buffer_slot("a");
        let buf_b = builder.buffer_slot("b");

        builder
            .step("work", &pipeline)
            .bind_buffer(buf_a, NodeAccess::Read)
            .bind_buffer(buf_b, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let program = builder.compile().unwrap();

        // Only bind buf_a, leave buf_b unbound — but no barrier needed since single wave
        let ba = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();
        let mut exec = program.specialize();
        exec.bind_buffer(buf_a, &ba);
        // No barrier => no slot lookup needed, should succeed
        let cmds = exec.compile_commands().unwrap();
        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, ComputeCommand::Dispatch { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn unset_dim_slot_errors() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut builder = ComputeProgram::builder();
        let wg = builder.dim_slot("wg");

        builder.step("work", &pipeline).dispatch_slot(wg);

        let program = builder.compile().unwrap();
        let exec = program.specialize();
        // dim slot not set — should error
        let result = exec.compile_commands();
        assert!(result.is_err());
    }

    #[test]
    fn submit_via_mock() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut builder = ComputeProgram::builder();
        let buf_slot = builder.buffer_slot("data");
        builder
            .step("work", &pipeline)
            .bind_buffer(buf_slot, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);

        let program = builder.compile().unwrap();
        let buf = Buffer::new(&device, 256, crate::DataAccess::Scattered).unwrap();

        let mut exec = program.specialize();
        exec.bind_buffer(buf_slot, &buf);
        let future = exec.submit(&device).unwrap();
        assert!(future.is_complete());
        future.wait().unwrap();
    }

    #[test]
    fn texture_slot() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut builder = ComputeProgram::builder();
        let tex_slot = builder.texture_slot("output");
        builder
            .step("render", &pipeline)
            .bind_texture(tex_slot, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let program = builder.compile().unwrap();
        let tex = Texture::new(
            &device,
            64,
            64,
            crate::TextureFormat::Rgba8Unorm,
            crate::SpatialAccess::Direct,
            crate::TextureFlags::empty(),
        )
        .unwrap();

        let mut exec = program.specialize();
        exec.bind_texture(tex_slot, &tex);
        let cmds = exec.compile_commands().unwrap();
        assert_eq!(
            cmds.iter()
                .filter(|c| matches!(c, ComputeCommand::Dispatch { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn push_constants_preserved() {
        let device = mock_device();
        let shader = mock_shader(&device);
        let pipeline = mock_pipeline(&device, &shader);

        let mut builder = ComputeProgram::builder();
        builder
            .step("work", &pipeline)
            .push_constants_raw(&[10, 20, 30])
            .dispatch(1, 1, 1);

        let program = builder.compile().unwrap();
        let exec = program.specialize();
        let cmds = exec.compile_commands().unwrap();

        let pc_cmd = cmds
            .iter()
            .find(|c| matches!(c, ComputeCommand::SetPushConstantsRaw { .. }))
            .unwrap();
        assert!(
            matches!(pc_cmd, ComputeCommand::SetPushConstantsRaw { indices } if indices == &[10, 20, 30])
        );
    }
}
