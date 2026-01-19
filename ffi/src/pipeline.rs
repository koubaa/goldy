//! FFI bindings for RenderPipeline.

use crate::device::GoldyDevice;
use crate::error::set_last_error_from_anyhow;
use crate::shader::GoldyShaderModule;
use crate::types::{
    GoldyCompareFunction, GoldyDepthFormat, GoldyPrimitiveTopology, GoldyTextureFormat,
    GoldyVertexAttribute,
};
use std::ptr;
use std::slice;

/// Opaque handle to a Goldy RenderPipeline.
pub struct GoldyRenderPipeline {
    pub(crate) inner: goldy::RenderPipeline,
}

/// Render pipeline descriptor for FFI.
#[repr(C)]
#[derive(Debug)]
pub struct GoldyRenderPipelineDesc {
    /// Pointer to vertex attributes array.
    pub vertex_attributes: *const GoldyVertexAttribute,
    /// Number of vertex attributes.
    pub vertex_attribute_count: u32,
    /// Stride in bytes between vertices.
    pub vertex_stride: u32,
    /// Primitive topology.
    pub topology: GoldyPrimitiveTopology,
    /// Target texture format.
    pub target_format: GoldyTextureFormat,
    /// Whether depth testing is enabled.
    pub depth_enabled: bool,
    /// Depth format (only used if depth_enabled is true).
    pub depth_format: GoldyDepthFormat,
    /// Whether to write depth values.
    pub depth_write_enabled: bool,
    /// Depth comparison function.
    pub depth_compare: GoldyCompareFunction,
}

impl Default for GoldyRenderPipelineDesc {
    fn default() -> Self {
        GoldyRenderPipelineDesc {
            vertex_attributes: ptr::null(),
            vertex_attribute_count: 0,
            vertex_stride: 24, // Default Vertex2D stride
            topology: GoldyPrimitiveTopology::TriangleList,
            target_format: GoldyTextureFormat::Rgba8Unorm,
            depth_enabled: false,
            depth_format: GoldyDepthFormat::Depth24Plus,
            depth_write_enabled: true,
            depth_compare: GoldyCompareFunction::Less,
        }
    }
}

/// Create a new render pipeline.
///
/// Returns a pointer to the pipeline, or null on failure.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_render_pipeline_create(
    device: *const GoldyDevice,
    vertex_shader: *const GoldyShaderModule,
    fragment_shader: *const GoldyShaderModule,
    desc: *const GoldyRenderPipelineDesc,
) -> *mut GoldyRenderPipeline {
    if device.is_null() || vertex_shader.is_null() || fragment_shader.is_null() || desc.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Null pointer in pipeline creation"));
        return ptr::null_mut();
    }

    let desc = &*desc;

    // Build vertex layout
    let attributes: Vec<goldy::VertexAttribute> =
        if desc.vertex_stride == 0 {
            // Vertex-less rendering (procedural geometry from SV_VertexID)
            vec![]
        } else if desc.vertex_attribute_count > 0 && !desc.vertex_attributes.is_null() {
            slice::from_raw_parts(desc.vertex_attributes, desc.vertex_attribute_count as usize)
                .iter()
                .map(|a| (*a).into())
                .collect()
        } else {
            // Default Vertex2DUv layout (position + uv)
            // This matches goldy::types::Vertex2DUv and FullscreenVertex shader input
            vec![
                goldy::VertexAttribute {
                    location: 0,
                    format: goldy::VertexFormat::Float32x2, // POSITION
                    offset: 0,
                },
                goldy::VertexAttribute {
                    location: 1,
                    format: goldy::VertexFormat::Float32x2, // TEXCOORD0 (UV)
                    offset: 8,
                },
            ]
        };

    let vertex_layout = goldy::VertexBufferLayout {
        stride: desc.vertex_stride,
        attributes,
    };

    // Build depth stencil state
    let depth_stencil = if desc.depth_enabled {
        Some(goldy::DepthStencilState {
            format: desc.depth_format.into(),
            depth_write_enabled: desc.depth_write_enabled,
            depth_compare: desc.depth_compare.into(),
        })
    } else {
        None
    };

    let pipeline_desc = goldy::RenderPipelineDesc {
        vertex_layout,
        topology: desc.topology.into(),
        target_format: desc.target_format.into(),
        depth_stencil,
    };

    match goldy::RenderPipeline::new(
        &(*device).inner,
        &(*vertex_shader).inner,
        &(*fragment_shader).inner,
        &pipeline_desc,
    ) {
        Ok(pipeline) => Box::into_raw(Box::new(GoldyRenderPipeline { inner: pipeline })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Destroy a render pipeline.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_render_pipeline_destroy(pipeline: *mut GoldyRenderPipeline) {
    if !pipeline.is_null() {
        drop(Box::from_raw(pipeline));
    }
}
