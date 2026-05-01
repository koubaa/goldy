//! Compute pipeline and pass management.

use crate::backend::{ComputeCommand, ComputePipelineHandle, GpuBackend};
use crate::buffer::Buffer;
use crate::device::Device;
use crate::shader::ShaderModule;
use crate::types::BindlessHandle;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// A compute pipeline.
///
/// Compute pipelines run compute shaders on the GPU, enabling general-purpose
/// GPU computing (GPGPU). They process data in parallel across many threads.
///
/// # Example
///
/// ```rust,no_run
/// use goldy::{Instance, DeviceType, ShaderModule, ComputePipeline};
///
/// let instance = Instance::new()?;
/// let device = instance.create_device(DeviceType::DiscreteGpu)?;
///
/// let shader = ShaderModule::from_slang(&device, r#"
///     #include "goldy_exp.slang"
///
///     struct PushConstants { uint buffer_idx; };
///     [[vk::push_constant]] PushConstants pc;
///
///     [shader("compute")]
///     [numthreads(64, 1, 1)]
///     void cs_main(uint3 id : SV_DispatchThreadID) {
///         // Access buffer via index
///         float val = asfloat(g_StorageBuffers[pc.buffer_idx].Load(id.x * 4));
///         g_StorageBuffers[pc.buffer_idx].Store(id.x * 4, asuint(val * 2.0));
///     }
/// "#)?;
///
/// let pipeline = ComputePipeline::new(&device, &shader)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct ComputePipeline {
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: ComputePipelineHandle,
}

impl ComputePipeline {
    /// Create a new compute pipeline.
    pub fn new(device: &Device, compute_shader: &ShaderModule) -> Result<Self> {
        tracing::debug!("Creating compute pipeline");
        let mut backend = device.backend.lock().unwrap();

        let handle = backend.create_compute_pipeline(device.handle, compute_shader.handle)?;

        tracing::debug!("Compute pipeline created");

        Ok(Self {
            backend: Arc::clone(&device.backend),
            handle,
        })
    }

    /// GPU handle for building raw [`ComputeCommand`](crate::backend::ComputeCommand) streams
    /// (e.g. [`Device::submit_compute_commands`](crate::Device::submit_compute_commands)).
    #[inline]
    pub fn gpu_pipeline_handle(&self) -> ComputePipelineHandle {
        self.handle
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        tracing::trace!("Destroying compute pipeline");
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
/// use goldy::{Instance, DeviceType, ComputeEncoder};
///
/// let instance = Instance::new()?;
/// let device = instance.create_device(DeviceType::DiscreteGpu)?;
///
/// let mut encoder = ComputeEncoder::new();
/// let mut pass = encoder.begin_compute_pass();
/// // pass.set_pipeline(&pipeline);
/// // pass.set_push_constants(&[&buffer]);
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
        tracing::debug!(
            command_count = self.commands.len(),
            "Dispatching compute commands"
        );
        let mut backend = device.backend.lock().unwrap();
        backend.dispatch_compute(device.handle, &self.commands)
    }

    /// Submit the recorded compute commands without blocking.
    ///
    /// Returns a [`GpuFuture`](crate::GpuFuture) that can be polled via
    /// [`is_complete`](crate::GpuFuture::is_complete) or awaited via
    /// [`wait`](crate::GpuFuture::wait) / [`wait_timeout`](crate::GpuFuture::wait_timeout).
    pub fn submit(&self, device: &Device) -> Result<crate::GpuFuture> {
        let mut backend = device.backend.lock().unwrap();
        let token = backend.submit_compute(device.handle, &self.commands)?;
        Ok(crate::GpuFuture {
            backend: Arc::clone(&device.backend),
            device: device.handle,
            fence_token: token,
        })
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
        self.encoder
            .commands
            .push(ComputeCommand::SetPipeline(pipeline.handle));
    }

    /// Set push constants for resource binding (compute shaders).
    ///
    /// Buffer indices are passed via push/root constants. The shader accesses
    /// resources through global descriptor arrays indexed by these values.
    ///
    /// # Example
    /// ```ignore
    /// pass.set_push_constants(&[&particle_buffer, &params_buffer]);
    /// // In shader: g_StorageBuffers[getBufferIndex(0)] for particles
    /// // In shader: g_UniformBuffers[getBufferIndex(1)] for params
    /// ```
    pub fn set_push_constants(&mut self, buffers: &[&Buffer]) {
        self.encoder
            .commands
            .push(ComputeCommand::SetPushConstants {
                buffers: buffers.iter().map(|b| b.handle).collect(),
            });
    }

    /// Set push constants with raw u32 indices (for textures/samplers or mixed resources).
    ///
    /// **Prefer [`ComputePass::set_push_constants_typed`]** for new code — the
    /// raw form skips per-slot category validation, so binding a uniform-buffer
    /// index into a slot the shader reads via `goldy_dyn_buf_ro` will silently
    /// produce garbage reads rather than erroring at dispatch time.
    ///
    /// # Example
    /// ```ignore
    /// let tex_idx = texture.bindless_index().unwrap();
    /// pass.set_push_constants_raw(&[buf_idx_0, buf_idx_1, tex_idx]);
    /// ```
    pub fn set_push_constants_raw(&mut self, indices: &[u32]) {
        self.encoder
            .commands
            .push(ComputeCommand::SetPushConstantsRaw {
                indices: indices.to_vec(),
            });
    }

    /// Set push constants from typed [`BindlessHandle`]s.
    ///
    /// Each handle carries both the raw index and the resource's
    /// [`crate::types::BindlessCategory`]. At dispatch time
    /// the backend cross-checks these against the bound shader's reflection
    /// and returns an error if any slot's category disagrees with how the
    /// shader reads it (e.g. binding a
    /// [`BindlessCategory::Broadcast`](crate::types::BindlessCategory::Broadcast)
    /// handle to a slot accessed via `goldy_dyn_buf_ro`, which reads the
    /// storage-buffer pool). When the shader provides no expectation for a slot
    /// (e.g. computed slot indices that regex analysis can't resolve),
    /// validation is skipped for that slot.
    ///
    /// # Example
    /// ```ignore
    /// let uniforms = uniform_buf.bindless_handle().unwrap();    // Broadcast
    /// let output  = output_tex.bindless_handle().unwrap();      // StorageImage
    /// pass.set_push_constants_typed(&[uniforms, output]);
    /// ```
    pub fn set_push_constants_typed(&mut self, handles: &[BindlessHandle]) {
        self.encoder
            .commands
            .push(ComputeCommand::SetPushConstantsTyped {
                handles: handles.to_vec(),
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

    /// Indirect dispatch: workgroup counts read from buffer at offset.
    ///
    /// The buffer must contain 3 consecutive `u32` values (x, y, z) at the given
    /// byte offset. This allows the GPU to determine dispatch size from a prior
    /// compute pass (e.g. a setup shader that writes the counts).
    pub fn dispatch_indirect(&mut self, buffer: &Buffer, offset: u64) {
        self.encoder
            .commands
            .push(ComputeCommand::DispatchIndirect {
                buffer: buffer.handle,
                offset,
            });
    }

    /// Insert a memory barrier between compute dispatches.
    ///
    /// Ensures all prior shader writes complete and are visible before
    /// any subsequent shader reads or writes execute.
    pub fn barrier(&mut self) {
        self.encoder.commands.push(ComputeCommand::Barrier);
    }

    /// Fill a buffer region with zeros, batched into the compute command stream.
    ///
    /// Unlike `Buffer::clear()` which submits immediately, this records the clear
    /// into the encoder so it's submitted alongside dispatches in a single batch.
    /// If `size` is 0, clears from `offset` to end of buffer.
    pub fn clear_buffer(&mut self, buffer: &Buffer, offset: u64, size: u64) {
        self.encoder.commands.push(ComputeCommand::ClearBuffer {
            buffer: buffer.handle,
            offset,
            size,
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
            ComputeCommand::Dispatch {
                workgroups_x,
                workgroups_y,
                workgroups_z,
            } => {
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
