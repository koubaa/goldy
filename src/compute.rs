//! Compute pipeline management.

use crate::backend::{ComputePipelineHandle, GpuBackend};
use crate::device::Device;
use crate::shader::ShaderModule;
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
/// use goldy::{ComputePipeline, DeviceDescriptor, Instance, RequestAdapterOptions, ShaderModule, TaskGraph};
///
/// let instance = Instance::new()?;
/// let device = instance
///     .request_adapter(&RequestAdapterOptions::default())?
///     .request_device(&DeviceDescriptor::default())?;
///
/// let shader = ShaderModule::from_slang(&device, r#"
///     import goldy_exp;
///
///     [goldy_compute]
///     [numthreads(64, 1, 1)]
///     void cs_main(Scattered<float> data, ThreadId id) {
///         data[id.x] = data[id.x] * 2.0;
///     }
/// "#)?;
///
/// let pipeline = ComputePipeline::new(&device, &shader)?;
/// let mut graph = TaskGraph::new();
/// graph.node("main", &pipeline).dispatch(1, 1, 1);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct ComputePipeline {
    _device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: ComputePipelineHandle,
    /// Per push-constant resource slot (shader-signature order), the descriptor
    /// access the shader signature requires. Used by [`crate::Scheme`] recording to
    /// pick the correct SRV/UAV descriptor independent of the graph access (which
    /// only drives barriers). Empty when the backend reports no reflection.
    pub(crate) slot_access: Vec<Option<crate::types::ResourceAccess>>,
}

impl ComputePipeline {
    /// Create a new compute pipeline.
    pub fn new(device: &Device, compute_shader: &ShaderModule) -> Result<Self> {
        tracing::debug!("Creating compute pipeline");

        // Pre-warm compilation of the compute shader's bytecode using a dedicated
        // shader-compilation lock instead of the backend's exclusive per-device lock, so a
        // slow Slang compile here doesn't stall every other thread's backend calls (submits,
        // buffer ops, waits, other shader compiles) for the duration of this compile.
        let precompile_prep = {
            let backend = device.inner.backend.lock().unwrap();
            backend.prepare_shader_stage_precompile(compute_shader.handle, crate::slang::SlangStage::Compute)?
        };
        if let Some(prep) = precompile_prep {
            let compiled = prep.compile()?;
            let mut backend = device.inner.backend.lock().unwrap();
            backend.store_precompiled_shader_stage(compute_shader.handle, crate::slang::SlangStage::Compute, compiled)?;
        }

        let mut backend = device.inner.backend.lock().unwrap();

        let handle = {
            let _tz = crate::tracy_zone!("goldy.compute_pipeline.create_pso");
            backend.create_compute_pipeline(device.inner.handle, compute_shader.handle)?
        };

        tracing::debug!("Compute pipeline created");

        let slot_access = backend.compute_pipeline_slot_access(handle);

        Ok(Self {
            _device: device.clone(),
            backend: Arc::clone(&device.inner.backend),
            handle,
            slot_access,
        })
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        tracing::trace!("Destroying compute pipeline");
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_compute_pipeline(self.handle);
    }
}
