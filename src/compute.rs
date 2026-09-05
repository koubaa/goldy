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
/// use goldy::{ComputePipeline, Context, DeviceDescriptor, Instance, RequestAdapterOptions, Scheme, ShaderModule};
///
/// let instance = Instance::new()?;
/// let device = instance
///     .request_adapter(&RequestAdapterOptions::default())?
///     .request_device(&DeviceDescriptor::default())?;
/// let ctx = device.create_context()?;
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
/// let mut scheme = Scheme::new(&ctx);
/// scheme.node("main", &pipeline).dispatch(1, 1, 1);
/// scheme.submit()?;
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
    /// Compile inputs of the shader this pipeline was built from, so a scheme can compile
    /// specialized variants of it without the caller keeping the [`ShaderModule`] alive.
    pub(crate) provenance: Arc<crate::shader::ShaderProvenance>,
}

impl ComputePipeline {
    /// Create a new compute pipeline.
    pub fn new(device: &Device, compute_shader: &ShaderModule) -> Result<Self> {
        Self::new_with_label(device, compute_shader, None)
    }

    /// Create a compute pipeline with an optional debug label for GPU tools.
    ///
    /// On Metal the label is applied to the `MTLFunction` and compute PSO so
    /// Instruments / Xcode Metal Debugger can distinguish shaders (e.g. `"fine_area"`
    /// instead of a generic `cs_main`). Other backends store it for CPU-side
    /// diagnostics.
    pub fn new_with_label(device: &Device, compute_shader: &ShaderModule, label: Option<&str>) -> Result<Self> {
        tracing::debug!(?label, "Creating compute pipeline");

        let seeded = {
            #[cfg(any(feature = "vulkan", feature = "dx12"))]
            let _st = crate::shader_timing::scope("compute.slang_unlocked", label.unwrap_or(""));
            compile_compute_stage_unlocked(device, compute_shader)?
        };

        let mut backend = device.inner.backend.lock().unwrap();
        if let Some((bytecode, reflection)) = seeded {
            backend.seed_compute_stage(compute_shader.handle, &bytecode, reflection)?;
        }

        let handle = {
            let _tz = crate::tracy_zone!("goldy.compute_pipeline.create_pso");
            backend.create_compute_pipeline(device.inner.handle, compute_shader.handle, label)?
        };

        tracing::debug!("Compute pipeline created");

        let slot_access = backend.compute_pipeline_slot_access(handle);

        Ok(Self {
            _device: device.clone(),
            backend: Arc::clone(&device.inner.backend),
            handle,
            slot_access,
            provenance: Arc::clone(compute_shader.provenance()),
        })
    }
}

/// Compile the compute stage without holding `device.inner.backend`.
///
/// Returns `None` when the backend has no Slang compute target (mock, CPU, Metal/WebGPU/CUDA
/// until those implement [`GpuBackend::seed_compute_stage`]) or when a frontend Slang session
/// cannot be created; PSO creation then compiles under the lock as before.
fn compile_compute_stage_unlocked(
    device: &Device,
    shader: &ShaderModule,
) -> Result<Option<(Vec<u8>, crate::slang::ShaderReflection)>> {
    let target = {
        let backend = device.inner.backend.lock().unwrap();
        backend.compute_shader_target()
    };
    let Some(target) = target else {
        return Ok(None);
    };

    let compiler = match device.inner.slang.get() {
        Some(compiler) => Arc::clone(compiler),
        None => {
            let created = match crate::slang::SlangCompiler::new() {
                Ok(compiler) => Arc::new(compiler),
                Err(err) => {
                    tracing::debug!(
                        ?err,
                        "frontend Slang init failed; compute compile stays under the backend lock"
                    );
                    return Ok(None);
                }
            };
            let _ = device.inner.slang.set(Arc::clone(&created));
            device.inner.slang.get().cloned().unwrap_or(created)
        }
    };

    let path_refs: Vec<&str> = shader.search_paths().iter().map(|s| s.as_str()).collect();
    let mut extra_defines: Vec<(&str, &str)> = shader.defines().iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let caps = device.capabilities();
    if (caps.ray_query || caps.ray_tracing_pipelines) && extra_defines.iter().all(|(k, _)| *k != "GOLDY_RAY_QUERY") {
        extra_defines.push(("GOLDY_RAY_QUERY", "1"));
    }
    let result = compiler.compile_bindless_with_reflection_effective(
        shader.source(),
        shader.effective_source(),
        target,
        &[("cs_main", crate::slang::SlangStage::Compute)],
        &path_refs,
        &extra_defines,
        shader.layout_checks(),
        shader.optimization_level(),
    )?;

    let mut reflection = result.reflection;
    if reflection.push_constant_categories.is_empty() {
        reflection.push_constant_categories =
            crate::slang::virtual_main::extract_push_constant_categories(shader.source());
    }
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    if reflection.push_constant_slot_kinds.is_empty() {
        reflection.push_constant_slot_kinds =
            crate::slang::virtual_main::extract_push_constant_slot_kinds(shader.source());
    }

    Ok(Some((result.shader.data, reflection)))
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        tracing::trace!("Destroying compute pipeline");
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_compute_pipeline(self.handle);
    }
}
