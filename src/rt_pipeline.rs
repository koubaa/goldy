//! Ray-tracing pipelines (`TraceRays` / `DispatchRays`) with an internal SBT.

use crate::backend::{GpuBackend, GpuRayTracingPipelineDesc, RayTracingPipelineHandle};
use crate::device::Device;
use crate::shader::ShaderModule;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// Triangle-hit ray-tracing pipeline (one raygen, one miss, one closest-hit).
///
/// The shader-binding table is allocated and filled by the backend. Bind resources
/// on [`crate::Scheme::trace_rays`] the same way as a compute node (bindless slots
/// from the **raygen** signature).
pub struct RayTracingPipeline {
    _device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: RayTracingPipelineHandle,
    pub(crate) slot_access: Vec<Option<crate::types::ResourceAccess>>,
}

/// Shader modules for [`RayTracingPipeline::new`].
///
/// All three modules may be the same [`ShaderModule`] when raygen, miss, and
/// closest-hit live in one Slang source.
pub struct RayTracingPipelineDesc<'a> {
    /// `[goldy_raygen]` / `rgen_main` module.
    pub raygen: &'a ShaderModule,
    /// `[goldy_miss]` / `rmiss_main` module.
    pub miss: &'a ShaderModule,
    /// `[goldy_closesthit]` / `rchit_main` module.
    pub closest_hit: &'a ShaderModule,
}

impl RayTracingPipeline {
    /// Create an RT pipeline when [`crate::DeviceCapabilities::ray_tracing_pipelines`] is set.
    pub fn new(device: &Device, desc: &RayTracingPipelineDesc<'_>) -> Result<Self> {
        Self::new_with_label(device, desc, None)
    }

    /// [`Self::new`] with an optional GPU-debugger label.
    pub fn new_with_label(device: &Device, desc: &RayTracingPipelineDesc<'_>, label: Option<&str>) -> Result<Self> {
        anyhow::ensure!(
            device.capabilities().ray_tracing_pipelines,
            "this adapter does not support ray tracing pipelines (DeviceCapabilities::ray_tracing_pipelines)"
        );
        tracing::debug!(?label, "Creating ray tracing pipeline");
        let mut backend = device.inner.backend.lock().unwrap();
        let handle = backend.create_ray_tracing_pipeline(
            device.inner.handle,
            GpuRayTracingPipelineDesc {
                raygen: desc.raygen.handle,
                miss: desc.miss.handle,
                closest_hit: desc.closest_hit.handle,
            },
            label,
        )?;
        let slot_access = backend.ray_tracing_pipeline_slot_access(handle);
        Ok(Self {
            _device: device.clone(),
            backend: Arc::clone(&device.inner.backend),
            handle,
            slot_access,
        })
    }
}

impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        if let Ok(mut backend) = self.backend.lock() {
            backend.destroy_ray_tracing_pipeline(self.handle);
        }
    }
}
