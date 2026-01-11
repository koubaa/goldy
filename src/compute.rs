//! Compute pipeline and pass management.

use crate::backend::{ComputeCommand, ComputePipelineHandle, GpuBackend, BindGroupLayoutHandle};
use crate::bind_group::{BindGroup, BindGroupLayout};
use crate::device::Device;
use crate::shader::ShaderModule;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// Description for creating a compute pipeline.
#[derive(Clone, Default)]
pub struct ComputePipelineDesc<'a> {
    /// Bind group layouts used by this pipeline (optional).
    /// The order determines the set index (first = set 0, second = set 1, etc.)
    pub bind_group_layouts: &'a [&'a BindGroupLayout],
}

impl<'a> std::fmt::Debug for ComputePipelineDesc<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputePipelineDesc")
            .field("bind_group_layouts_count", &self.bind_group_layouts.len())
            .finish()
    }
}

/// A compute pipeline.
///
/// Compute pipelines run compute shaders on the GPU, enabling general-purpose
/// GPU computing (GPGPU). They process data in parallel across many threads.
///
/// # Example
///
/// ```rust,no_run
/// use rag::{Instance, DeviceType, ShaderModule, ComputePipeline, ComputePipelineDesc};
///
/// let instance = Instance::new()?;
/// let device = instance.create_device(DeviceType::DiscreteGpu)?;
///
/// let shader = ShaderModule::from_slang(&device, r#"
///     [[vk::binding(0, 0)]] RWStructuredBuffer<float> data;
///
///     [shader("compute")]
///     [numthreads(64, 1, 1)]
///     void cs_main(uint3 id : SV_DispatchThreadID) {
///         data[id.x] = data[id.x] * 2.0;
///     }
/// "#)?;
///
/// let pipeline = ComputePipeline::new(&device, &shader, &ComputePipelineDesc::default())?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct ComputePipeline {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: ComputePipelineHandle,
}

impl ComputePipeline {
    /// Create a new compute pipeline.
    pub fn new(
        device: &Device,
        compute_shader: &ShaderModule,
        desc: &ComputePipelineDesc,
    ) -> Result<Self> {
        let mut backend = device.backend.lock().unwrap();
        
        // Collect bind group layout handles
        let layout_handles: Vec<BindGroupLayoutHandle> = desc
            .bind_group_layouts
            .iter()
            .map(|l| l.handle)
            .collect();
        
        let handle = backend.create_compute_pipeline(
            device.handle,
            compute_shader.handle,
            &layout_handles,
        )?;

        Ok(Self {
            backend: Arc::clone(&device.backend),
            handle,
        })
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_compute_pipeline(self.handle);
    }
}

/// Command encoder for compute operations.
///
/// Similar to `CommandEncoder` for graphics, but for compute workloads.
/// Commands are recorded lock-free and executed when `dispatch()` is called.
///
/// # Example
///
/// ```rust,no_run
/// use rag::{Instance, DeviceType, ComputeEncoder};
///
/// let instance = Instance::new()?;
/// let device = instance.create_device(DeviceType::DiscreteGpu)?;
///
/// let mut encoder = ComputeEncoder::new();
/// let mut pass = encoder.begin_compute_pass();
/// // pass.set_pipeline(&pipeline);
/// // pass.set_bind_group(0, &bind_group);
/// // pass.dispatch(1, 1, 1);
/// drop(pass);
///
/// let commands = encoder.finish();
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct ComputeEncoder {
    pub(crate) commands: Vec<ComputeCommand>,
}

impl ComputeEncoder {
    /// Create a new compute encoder.
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Begin a compute pass.
    pub fn begin_compute_pass(&mut self) -> ComputePass<'_> {
        ComputePass { encoder: self }
    }

    /// Get the recorded commands.
    pub fn finish(self) -> Vec<ComputeCommand> {
        self.commands
    }

    /// Execute the recorded compute commands on the device.
    ///
    /// This submits the compute work to the GPU and waits for completion.
    pub fn dispatch(&self, device: &Device) -> Result<()> {
        let mut backend = device.backend.lock().unwrap();
        backend.dispatch_compute(device.handle, &self.commands)
    }
}

impl Default for ComputeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// A compute pass for recording compute operations.
pub struct ComputePass<'a> {
    encoder: &'a mut ComputeEncoder,
}

impl<'a> ComputePass<'a> {
    /// Set the active compute pipeline.
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        self.encoder.commands.push(ComputeCommand::SetPipeline(pipeline.handle));
    }

    /// Set a bind group for shader resources (uniforms, storage buffers).
    ///
    /// The `index` corresponds to the bind group set in the shader
    /// (e.g., `[[vk::binding(0, 0)]]` uses index 0).
    pub fn set_bind_group(&mut self, index: u32, bind_group: &BindGroup) {
        self.encoder.commands.push(ComputeCommand::SetBindGroup {
            index,
            bind_group: bind_group.handle,
        });
    }

    /// Dispatch compute workgroups.
    ///
    /// This records a dispatch command with the specified number of workgroups
    /// in each dimension. The actual number of threads is:
    /// `workgroups * numthreads` (as specified in the shader's `[numthreads]` attribute).
    ///
    /// # Example
    ///
    /// For a shader with `[numthreads(64, 1, 1)]`:
    /// - `dispatch(16, 1, 1)` runs 16 * 64 = 1024 threads
    /// - `dispatch(256, 1, 1)` runs 256 * 64 = 16384 threads
    pub fn dispatch(&mut self, workgroups_x: u32, workgroups_y: u32, workgroups_z: u32) {
        self.encoder.commands.push(ComputeCommand::Dispatch {
            workgroups_x,
            workgroups_y,
            workgroups_z,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_encoder_creation() {
        let encoder = ComputeEncoder::new();
        assert!(encoder.commands.is_empty());
    }

    #[test]
    fn test_compute_encoder_default() {
        let encoder = ComputeEncoder::default();
        assert!(encoder.commands.is_empty());
    }

    #[test]
    fn test_dispatch_command() {
        let mut encoder = ComputeEncoder::new();
        {
            let mut pass = encoder.begin_compute_pass();
            pass.dispatch(4, 2, 1);
        }
        let commands = encoder.finish();
        
        assert_eq!(commands.len(), 1);
        match &commands[0] {
            ComputeCommand::Dispatch { workgroups_x, workgroups_y, workgroups_z } => {
                assert_eq!(*workgroups_x, 4);
                assert_eq!(*workgroups_y, 2);
                assert_eq!(*workgroups_z, 1);
            }
            _ => panic!("Expected Dispatch command"),
        }
    }

    #[test]
    fn test_multiple_dispatches() {
        let mut encoder = ComputeEncoder::new();
        {
            let mut pass = encoder.begin_compute_pass();
            pass.dispatch(1, 1, 1);
            pass.dispatch(8, 8, 1);
            pass.dispatch(256, 1, 1);
        }
        let commands = encoder.finish();
        
        assert_eq!(commands.len(), 3);
        assert!(matches!(&commands[0], ComputeCommand::Dispatch { .. }));
        assert!(matches!(&commands[1], ComputeCommand::Dispatch { .. }));
        assert!(matches!(&commands[2], ComputeCommand::Dispatch { .. }));
    }
}

