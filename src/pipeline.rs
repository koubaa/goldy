//! Render pipeline management.

use crate::backend::{GpuBackend, PipelineHandle};
use crate::device::Device;
use crate::shader::ShaderModule;
use crate::slang::ffi::SlangStage;
use crate::slang::graphics_link::{
    remap_is_identity, GraphicsPipelineInterface, LinkedMeshPipeline, LinkedRasterPipeline, PipelineResourceContract,
    SlotRemap,
};
use crate::slang::virtual_main::Stage;
use crate::types::*;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Description for creating a render pipeline.
///
/// # Format Matching
///
/// **Important**: `target_format` must match the format of the render target
/// you will render to. Mismatched formats cause undefined behavior or errors.
///
/// - For scheme-leased render targets: use the format passed to [`crate::Scheme::lease_render_target`]
///
/// # Example
///
/// ```rust,no_run
/// # use goldy::{RenderPipelineDesc, TextureFormat};
/// let desc = RenderPipelineDesc {
///     target_format: TextureFormat::Rgba8Unorm,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Default, Debug)]
pub struct RenderPipelineDesc {
    /// Vertex buffer layout.
    pub vertex_layout: VertexBufferLayout,
    /// Primitive topology.
    pub topology: PrimitiveTopology,
    /// Target texture format.
    ///
    /// **Must match** the format of the swapchain or scheme-leased render target you render to.
    /// Use the format passed to [`crate::Scheme::lease_render_target`].
    pub target_format: TextureFormat,
    /// Depth/stencil state (optional, None = no depth testing).
    pub depth_stencil: Option<DepthStencilState>,
}

impl Default for VertexBufferLayout {
    /// Returns an empty layout (no vertex attributes, stride 0).
    ///
    /// Use this for passes whose vertex shader generates geometry from `SV_VertexID`
    /// / `VertexId` without reading a vertex buffer (fullscreen quads, procedural
    /// triangles, etc.). For typed vertex input use `Vertex2D::layout()` or
    /// `VertexBufferLayout::from_formats::<T>(&[…])` explicitly.
    fn default() -> Self {
        Self {
            stride: 0,
            attributes: Vec::new(),
        }
    }
}

/// A render pipeline.
pub struct RenderPipeline {
    _device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: PipelineHandle,
    /// Per push-constant resource slot (merged pipeline contract order).
    pub(crate) slot_access: Vec<Option<crate::types::ResourceAccess>>,
    pub(crate) resource_contract: PipelineResourceContract,
    interface: GraphicsPipelineInterface,
}

impl RenderPipeline {
    /// Create a new render pipeline.
    ///
    /// Existing constructors link `[goldy_vertex]` / `[goldy_fragment]` stages automatically.
    /// Prefer [`Self::builder`] when teaching the vertex → payload → fragment model.
    pub fn new(
        device: &Device,
        vertex_shader: &ShaderModule,
        fragment_shader: &ShaderModule,
        desc: &RenderPipelineDesc,
    ) -> Result<Self> {
        Self::builder(device)
            .vertex(vertex_shader)
            .fragment(fragment_shader)
            .vertex_layout(desc.vertex_layout.clone())
            .topology(desc.topology)
            .target_format(desc.target_format)
            .depth_stencil(desc.depth_stencil.clone())
            .build()
    }

    /// Start a pipeline builder that reflects and links stage I/O.
    pub fn builder(device: &Device) -> RenderPipelineBuilder<'_> {
        RenderPipelineBuilder {
            device,
            vertex: None,
            fragment: None,
            desc: RenderPipelineDesc::default(),
        }
    }

    /// Reflected vertex input, payload links, fragment outputs, and resource contract.
    pub fn interface(&self) -> &GraphicsPipelineInterface {
        &self.interface
    }

    pub(crate) fn resource_contract(&self) -> &PipelineResourceContract {
        &self.resource_contract
    }
}

/// Additive builder for [`RenderPipeline`].
pub struct RenderPipelineBuilder<'a> {
    device: &'a Device,
    vertex: Option<&'a ShaderModule>,
    fragment: Option<&'a ShaderModule>,
    desc: RenderPipelineDesc,
}

impl<'a> RenderPipelineBuilder<'a> {
    pub fn vertex(mut self, shader: &'a ShaderModule) -> Self {
        self.vertex = Some(shader);
        self
    }

    pub fn fragment(mut self, shader: &'a ShaderModule) -> Self {
        self.fragment = Some(shader);
        self
    }

    pub fn vertex_layout(mut self, layout: VertexBufferLayout) -> Self {
        self.desc.vertex_layout = layout;
        self
    }

    pub fn topology(mut self, topology: PrimitiveTopology) -> Self {
        self.desc.topology = topology;
        self
    }

    pub fn target_format(mut self, format: TextureFormat) -> Self {
        self.desc.target_format = format;
        self
    }

    pub fn depth_stencil(mut self, state: Option<DepthStencilState>) -> Self {
        self.desc.depth_stencil = state;
        self
    }

    pub fn build(self) -> Result<RenderPipeline> {
        let vertex_shader = self
            .vertex
            .ok_or_else(|| anyhow::anyhow!("RenderPipeline::builder requires .vertex(&shader)"))?;
        let fragment_shader = self
            .fragment
            .ok_or_else(|| anyhow::anyhow!("RenderPipeline::builder requires .fragment(&shader)"))?;
        let desc = &self.desc;

        tracing::debug!(
            target_format = ?desc.target_format,
            topology = ?desc.topology,
            has_depth = desc.depth_stencil.is_some(),
            "Creating render pipeline"
        );

        let mut linked = crate::slang::graphics_link::link_raster_pipeline(
            &vertex_shader.source,
            &fragment_shader.source,
            Some(&desc.vertex_layout),
        )?;

        let mut backend = self.device.inner.backend.lock().unwrap();
        if let Some(ref mut link) = linked {
            apply_stage_remap(
                &mut **backend,
                vertex_shader,
                SlangStage::Vertex,
                Stage::Vertex,
                &link.vs_remap,
            );
            apply_stage_remap(
                &mut **backend,
                fragment_shader,
                SlangStage::Fragment,
                Stage::Fragment,
                &link.fs_remap,
            );
            let _ = backend.compile_shader_stage(vertex_shader.handle, SlangStage::Vertex);
            let _ = backend.compile_shader_stage(fragment_shader.handle, SlangStage::Fragment);
            refine_raster_from_reflection(&**backend, vertex_shader, fragment_shader, &desc.vertex_layout, link)?;
        }

        let handle = if desc.depth_stencil.is_some() {
            backend.create_pipeline_with_depth(
                self.device.inner.handle,
                vertex_shader.handle,
                fragment_shader.handle,
                &desc.vertex_layout,
                desc.topology,
                desc.target_format,
                desc.depth_stencil.as_ref(),
            )?
        } else {
            backend.create_pipeline(
                self.device.inner.handle,
                vertex_shader.handle,
                fragment_shader.handle,
                &desc.vertex_layout,
                desc.topology,
                desc.target_format,
            )?
        };

        let mut slot_access = backend.render_pipeline_slot_access(handle);
        let mut interface = GraphicsPipelineInterface::default();
        let mut resource_contract = PipelineResourceContract::default();
        if let Some(link) = linked {
            backend.apply_graphics_resource_contract(handle, &link.interface.resources);
            slot_access = link.interface.resources.slot_access();
            resource_contract = link.interface.resources.clone();
            interface = link.interface;
        }

        tracing::debug!("Render pipeline created");

        Ok(RenderPipeline {
            _device: self.device.clone(),
            backend: Arc::clone(&self.device.inner.backend),
            handle,
            slot_access,
            resource_contract,
            interface,
        })
    }
}

impl Drop for RenderPipeline {
    fn drop(&mut self) {
        tracing::trace!("Destroying render pipeline");
        let mut backend = self.backend.lock().unwrap();
        backend.destroy_pipeline(self.handle);
    }
}

/// Mesh (+ optional amplification) graphics pipeline.
///
/// Replaces the vertex stage with a mesh shader. Record draws with
/// [`crate::SchemeRenderPassBuilder::set_mesh_pipeline`] and
/// [`crate::SchemeRenderPassBuilder::dispatch_mesh`].
pub struct MeshPipeline {
    _device: Device,
    backend: Arc<Mutex<Box<dyn GpuBackend>>>,
    pub(crate) handle: PipelineHandle,
    pub(crate) slot_access: Vec<Option<crate::types::ResourceAccess>>,
    pub(crate) resource_contract: PipelineResourceContract,
    interface: GraphicsPipelineInterface,
}

/// Shader modules and raster state for [`MeshPipeline::new`].
pub struct MeshPipelineDesc<'a> {
    /// `[goldy_mesh]` / `mesh_main` module.
    pub mesh: &'a ShaderModule,
    /// `[goldy_fragment]` / `fs_main` module.
    pub fragment: &'a ShaderModule,
    /// Optional `[goldy_amplification]` / `amp_main` module.
    pub amplification: Option<&'a ShaderModule>,
    /// Pixel format of the render target — must match the leased target.
    pub target_format: TextureFormat,
    /// Depth/stencil state (optional, None = no depth testing).
    pub depth_stencil: Option<DepthStencilState>,
}

impl MeshPipeline {
    /// Create a mesh pipeline when [`crate::DeviceCapabilities::mesh_shaders`] is set.
    pub fn new(device: &Device, desc: &MeshPipelineDesc<'_>) -> Result<Self> {
        Self::builder(device)
            .mesh(desc.mesh)
            .fragment(desc.fragment)
            .amplification(desc.amplification)
            .target_format(desc.target_format)
            .depth_stencil(desc.depth_stencil.clone())
            .build()
    }

    /// Start a mesh pipeline builder that reflects and links stage I/O.
    pub fn builder(device: &Device) -> MeshPipelineBuilder<'_> {
        MeshPipelineBuilder {
            device,
            mesh: None,
            fragment: None,
            amplification: None,
            target_format: TextureFormat::default(),
            depth_stencil: None,
            label: None,
        }
    }

    /// [`Self::new`] with an optional GPU-debugger label.
    pub fn new_with_label(device: &Device, desc: &MeshPipelineDesc<'_>, label: Option<&str>) -> Result<Self> {
        Self::builder(device)
            .mesh(desc.mesh)
            .fragment(desc.fragment)
            .amplification(desc.amplification)
            .target_format(desc.target_format)
            .depth_stencil(desc.depth_stencil.clone())
            .label(label)
            .build()
    }

    pub fn interface(&self) -> &GraphicsPipelineInterface {
        &self.interface
    }

    pub(crate) fn resource_contract(&self) -> &PipelineResourceContract {
        &self.resource_contract
    }
}

/// Additive builder for [`MeshPipeline`].
pub struct MeshPipelineBuilder<'a> {
    device: &'a Device,
    mesh: Option<&'a ShaderModule>,
    fragment: Option<&'a ShaderModule>,
    amplification: Option<&'a ShaderModule>,
    target_format: TextureFormat,
    depth_stencil: Option<DepthStencilState>,
    label: Option<&'a str>,
}

impl<'a> MeshPipelineBuilder<'a> {
    pub fn mesh(mut self, shader: &'a ShaderModule) -> Self {
        self.mesh = Some(shader);
        self
    }

    pub fn fragment(mut self, shader: &'a ShaderModule) -> Self {
        self.fragment = Some(shader);
        self
    }

    pub fn amplification(mut self, shader: Option<&'a ShaderModule>) -> Self {
        self.amplification = shader;
        self
    }

    pub fn target_format(mut self, format: TextureFormat) -> Self {
        self.target_format = format;
        self
    }

    pub fn depth_stencil(mut self, state: Option<DepthStencilState>) -> Self {
        self.depth_stencil = state;
        self
    }

    pub fn label(mut self, label: Option<&'a str>) -> Self {
        self.label = label;
        self
    }

    pub fn build(self) -> Result<MeshPipeline> {
        let mesh = self
            .mesh
            .ok_or_else(|| anyhow::anyhow!("MeshPipeline::builder requires .mesh(&shader)"))?;
        let fragment = self
            .fragment
            .ok_or_else(|| anyhow::anyhow!("MeshPipeline::builder requires .fragment(&shader)"))?;
        anyhow::ensure!(
            self.device.capabilities().mesh_shaders,
            "this adapter does not support mesh shaders (DeviceCapabilities::mesh_shaders is false). \
             hint: skip MeshPipeline::new, or pick an adapter with mesh shaders \
             (Vulkan VK_EXT_mesh_shader, DX12 mesh tier 1, Metal Apple7 / Mac2). \
             Query device.capabilities().mesh_shaders."
        );
        if self.amplification.is_some() {
            anyhow::ensure!(
                self.device.capabilities().amplification_shaders,
                "this adapter does not support amplification/task shaders \
                 (DeviceCapabilities::amplification_shaders is false). \
                 hint: set MeshPipelineDesc::amplification to None, or pick an adapter that \
                 reports amplification_shaders."
            );
        }
        tracing::debug!(?self.label, "Creating mesh pipeline");

        let mut linked = crate::slang::graphics_link::link_mesh_pipeline(
            &mesh.source,
            &fragment.source,
            self.amplification.map(|s| s.source.as_ref()),
        )?;

        let vertex_layout = crate::types::VertexBufferLayout::default();
        let raster = crate::backend::shared::PipelineDesc::new(
            &vertex_layout,
            crate::types::PrimitiveTopology::TriangleList,
            self.target_format,
        );
        let mut backend = self.device.inner.backend.lock().unwrap();
        if let Some(ref mut link) = linked {
            apply_stage_remap(&mut **backend, mesh, SlangStage::Mesh, Stage::Mesh, &link.mesh_remap);
            apply_stage_remap(
                &mut **backend,
                fragment,
                SlangStage::Fragment,
                Stage::Fragment,
                &link.fs_remap,
            );
            if let Some(amp) = self.amplification {
                apply_stage_remap(
                    &mut **backend,
                    amp,
                    SlangStage::Amplification,
                    Stage::Amplification,
                    &link.amp_remap,
                );
                let _ = backend.compile_shader_stage(amp.handle, SlangStage::Amplification);
            }
            let _ = backend.compile_shader_stage(mesh.handle, SlangStage::Mesh);
            let _ = backend.compile_shader_stage(fragment.handle, SlangStage::Fragment);
            refine_mesh_from_reflection(&**backend, mesh, fragment, self.amplification, link)?;
        }
        let handle = backend.create_mesh_pipeline(
            self.device.inner.handle,
            crate::backend::GpuMeshPipelineDesc {
                mesh: mesh.handle,
                fragment: fragment.handle,
                amplification: self.amplification.map(|s| s.handle),
            },
            &raster,
            self.depth_stencil.as_ref(),
            self.label,
        )?;
        let mut slot_access = backend.render_pipeline_slot_access(handle);
        let mut interface = GraphicsPipelineInterface::default();
        let mut resource_contract = PipelineResourceContract::default();
        if let Some(link) = linked {
            backend.apply_graphics_resource_contract(handle, &link.interface.resources);
            slot_access = link.interface.resources.slot_access();
            resource_contract = link.interface.resources.clone();
            interface = link.interface;
        }
        Ok(MeshPipeline {
            _device: self.device.clone(),
            backend: Arc::clone(&self.device.inner.backend),
            handle,
            slot_access,
            resource_contract,
            interface,
        })
    }
}

impl Drop for MeshPipeline {
    fn drop(&mut self) {
        tracing::trace!("Destroying mesh pipeline");
        if let Ok(mut backend) = self.backend.lock() {
            backend.destroy_pipeline(self.handle);
        }
    }
}

fn apply_stage_remap(
    backend: &mut dyn GpuBackend,
    shader: &ShaderModule,
    slang_stage: SlangStage,
    vm_stage: Stage,
    remap: &SlotRemap,
) {
    let local_names: Vec<String> = crate::slang::virtual_main::find_all_entries(&shader.source)
        .into_iter()
        .find(|e| e.stage == vm_stage)
        .map(|e| {
            crate::slang::virtual_main::flatten_bindless_params(&e.params)
                .into_iter()
                .map(|p| p.name)
                .collect()
        })
        .unwrap_or_default();
    if remap_is_identity(&local_names, remap) {
        // Clear a leftover remap from another pipeline that shares this module.
        backend.set_shader_stage_slot_remap(shader.handle, slang_stage, HashMap::new());
        return;
    }
    backend.set_shader_stage_slot_remap(shader.handle, slang_stage, remap.clone());
}

fn refine_raster_from_reflection(
    backend: &dyn GpuBackend,
    vertex_shader: &ShaderModule,
    fragment_shader: &ShaderModule,
    vertex_layout: &VertexBufferLayout,
    link: &mut LinkedRasterPipeline,
) -> Result<()> {
    let Some(vs_io) = backend.shader_stage_interface(vertex_shader.handle, SlangStage::Vertex) else {
        return Ok(());
    };
    let Some(fs_io) = backend.shader_stage_interface(fragment_shader.handle, SlangStage::Fragment) else {
        return Ok(());
    };
    if vs_io.payload_outputs.is_empty() && fs_io.payload_inputs.is_empty() && vs_io.vertex_inputs.is_empty() {
        return Ok(());
    }
    if !vs_io.vertex_inputs.is_empty() {
        crate::slang::graphics_link::validate_vertex_layout(vertex_layout, &vs_io.vertex_inputs)?;
        link.interface.vertex_input = vs_io.vertex_inputs.clone();
    }
    if !vs_io.payload_outputs.is_empty() || !fs_io.payload_inputs.is_empty() {
        link.interface.payload_links =
            crate::slang::graphics_link::refine_payload_link("vertex", "fragment", &vs_io, &fs_io)?;
    }
    if !fs_io.fragment_outputs.is_empty() {
        link.interface.fragment_outputs = fs_io.fragment_outputs;
    }
    Ok(())
}

fn refine_mesh_from_reflection(
    backend: &dyn GpuBackend,
    mesh: &ShaderModule,
    fragment: &ShaderModule,
    amp: Option<&ShaderModule>,
    link: &mut LinkedMeshPipeline,
) -> Result<()> {
    let Some(mesh_io) = backend.shader_stage_interface(mesh.handle, SlangStage::Mesh) else {
        return Ok(());
    };
    let Some(fs_io) = backend.shader_stage_interface(fragment.handle, SlangStage::Fragment) else {
        return Ok(());
    };
    if looks_like_interpolated_payload(&mesh_io.payload_outputs) && !fs_io.payload_inputs.is_empty() {
        link.interface.payload_links =
            crate::slang::graphics_link::refine_payload_link("mesh", "fragment", &mesh_io, &fs_io)?;
    } else if mesh_io.payload_outputs.is_empty()
        && looks_like_interpolated_payload(&mesh_io.payload_inputs)
        && !fs_io.payload_inputs.is_empty()
    {
        let mut producer = mesh_io.clone();
        producer.payload_outputs = mesh_io.payload_inputs.clone();
        link.interface.payload_links =
            crate::slang::graphics_link::refine_payload_link("mesh", "fragment", &producer, &fs_io)?;
    }
    if let Some(amp) = amp {
        if let Some(amp_io) = backend.shader_stage_interface(amp.handle, SlangStage::Amplification) {
            let _ = crate::slang::graphics_link::refine_payload_link("amplification", "mesh", &amp_io, &mesh_io)?;
        }
    }
    if !fs_io.fragment_outputs.is_empty() {
        link.interface.fragment_outputs = fs_io.fragment_outputs;
    }
    Ok(())
}

fn looks_like_interpolated_payload(fields: &[crate::slang::graphics_link::StageIoField]) -> bool {
    fields.iter().any(|f| {
        let s = f.semantic.to_ascii_uppercase();
        s.starts_with("SV_POSITION") || s.starts_with("TEXCOORD") || s.starts_with("COLOR") || s == "POSITION"
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;

    fn create_test_device() -> Device {
        Device::from_backend(Box::new(MockBackend::new())).unwrap()
    }

    fn create_test_shader(device: &Device) -> ShaderModule {
        // Mock backend doesn't actually compile shaders, so any source works
        ShaderModule::from_slang(device, "mock shader source").unwrap()
    }

    #[test]
    fn test_render_pipeline_desc_default() {
        let desc = RenderPipelineDesc::default();

        assert_eq!(desc.target_format, TextureFormat::Rgba8Unorm);
        assert!(desc.depth_stencil.is_none());
    }

    #[test]
    fn test_render_pipeline_desc_debug() {
        let desc = RenderPipelineDesc {
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        };

        let debug_str = format!("{:?}", desc);
        assert!(debug_str.contains("RenderPipelineDesc"));
        assert!(debug_str.contains("Rgba8Unorm"));
    }

    #[test]
    fn test_simple_pipeline_creation() {
        let device = create_test_device();
        let shader = create_test_shader(&device);

        let pipeline = RenderPipeline::new(&device, &shader, &shader, &RenderPipelineDesc::default()).unwrap();

        assert!(pipeline.handle > 0);
    }

    #[test]
    fn test_pipeline_with_custom_format() {
        let device = create_test_device();
        let shader = create_test_shader(&device);

        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                target_format: TextureFormat::Bgra8UnormSrgb,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(pipeline.handle > 0);
    }

    #[test]
    fn test_pipeline_with_topology() {
        let device = create_test_device();
        let shader = create_test_shader(&device);

        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                topology: PrimitiveTopology::LineList,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(pipeline.handle > 0);
    }

    #[test]
    fn test_pipeline_with_depth_stencil() {
        let device = create_test_device();
        let shader = create_test_shader(&device);

        let pipeline = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                depth_stencil: Some(DepthStencilState {
                    format: DepthFormat::Depth24Plus,
                    depth_write_enabled: true,
                    depth_compare: CompareFunction::Less,
                }),
                ..Default::default()
            },
        )
        .unwrap();

        assert!(pipeline.handle > 0);
    }

    #[test]
    fn test_pipeline_with_vertex_layout() {
        use crate::types::Vertex2D;

        let device = create_test_device();
        let shader = create_test_shader(&device);

        // Test with Vertex2D layout
        let pipeline_2d = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                vertex_layout: Vertex2D::layout(),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(pipeline_2d.handle > 0);
    }

    #[test]
    fn test_pipeline_different_vertex_fragment_shaders() {
        let device = create_test_device();

        // In real use, these would be different shaders
        let vertex_shader = create_test_shader(&device);
        let fragment_shader = create_test_shader(&device);

        let pipeline = RenderPipeline::new(
            &device,
            &vertex_shader,
            &fragment_shader,
            &RenderPipelineDesc::default(),
        )
        .unwrap();

        assert!(pipeline.handle > 0);
    }

    #[test]
    fn test_multiple_pipelines() {
        let device = create_test_device();
        let shader = create_test_shader(&device);

        // Create multiple pipelines with different configurations
        let pipeline1 = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                target_format: TextureFormat::Rgba8Unorm,
                ..Default::default()
            },
        )
        .unwrap();

        let pipeline2 = RenderPipeline::new(
            &device,
            &shader,
            &shader,
            &RenderPipelineDesc {
                target_format: TextureFormat::Bgra8UnormSrgb,
                ..Default::default()
            },
        )
        .unwrap();

        // Pipelines should have different handles
        assert_ne!(pipeline1.handle, pipeline2.handle);
    }

    const LINKED_VS_FS: &str = r#"
struct VertIn { float2 pos : POSITION; };
struct Varying { float4 position : SV_Position; float2 uv : TEXCOORD0; };
[goldy_vertex]
Varying vs_main(VertexId vid) {
    Varying o;
    o.position = float4(0, 0, 0, 1);
    o.uv = float2(0, 0);
    return o;
}
[goldy_fragment]
float4 fs_main(Interpolated<float4> tex, Filter smp, Varying input) : SV_Target {
    return tex.Sample(smp, input.uv);
}
"#;

    #[test]
    fn builder_links_stage_local_resources() {
        let device = create_test_device();
        let shader = ShaderModule::from_slang(&device, LINKED_VS_FS).unwrap();
        let pipeline = RenderPipeline::builder(&device)
            .vertex(&shader)
            .fragment(&shader)
            .build()
            .unwrap();
        let names: Vec<&str> = pipeline
            .interface()
            .resources
            .resources
            .iter()
            .map(|r| r.name.as_str())
            .collect();
        assert_eq!(names, vec!["tex", "smp"]);
        assert_eq!(pipeline.interface().payload_links.len(), 2);
    }

    #[test]
    fn test_mesh_pipeline_creation() {
        let device = create_test_device();
        let shader = create_test_shader(&device);
        let pipeline = MeshPipeline::new(
            &device,
            &MeshPipelineDesc {
                mesh: &shader,
                fragment: &shader,
                amplification: None,
                target_format: TextureFormat::Rgba8Unorm,
                depth_stencil: None,
            },
        )
        .unwrap();
        assert!(pipeline.handle > 0);
    }
}
