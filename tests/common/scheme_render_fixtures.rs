//! Scheme render fixtures for FLIP screenshot tests and scheme render integration.

use goldy::{
    types::ResourceAccess, BufferKind, Color, CompareFunction, ComputePipeline, DepthFormat, DepthStencilState, Device,
    Instance, NodeAccess, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RenderTarget, RequestAdapterOptions,
    Scheme, ShaderModule, ShaderResourceSlot, TextureFormat, Vertex2D, VertexAttribute, VertexBufferLayout,
    VertexFormat,
};
use std::sync::Arc;

use super::scheme_render::{acquire_readback_texture, scheme_render_and_readback};
use crate::render_fixtures::{create_gol_initial_state, GOL_GRID_HEIGHT, GOL_GRID_WIDTH};

pub fn create_device() -> Option<Device> {
    let instance = Instance::new().ok()?;
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .ok()?
        .request_device(&goldy::DeviceDescriptor::default())
        .ok()
}

pub fn scheme_render_clear(device: &Device, width: u32, height: u32, color: Color) -> Vec<u8> {
    let ctx = device.create_context().expect("context");
    let target =
        RenderTarget::new(device, width, height, TextureFormat::Rgba8Unorm).expect("Failed to create render target");
    let mut pool = goldy::RetainedPool::new(Arc::new(device.clone()));
    let readback = acquire_readback_texture(&mut pool, width, height, TextureFormat::Rgba8Unorm);
    scheme_render_and_readback(&ctx, &target, &readback, "clear", |pass| {
        pass.clear(color);
    })
}

pub fn scheme_render_triangle(
    device: &Device,
    width: u32,
    height: u32,
    clear_color: Color,
    vertices: [Vertex2D; 3],
) -> Vec<u8> {
    let ctx = device.create_context().expect("context");
    let target =
        RenderTarget::new(device, width, height, TextureFormat::Rgba8Unorm).expect("Failed to create render target");

    let shader_source = r#"
        struct VertexInput {
            float2 position : POSITION;
            float4 color : COLOR;
        };

        struct VertexOutput {
            float4 position : SV_Position;
            float4 color : COLOR;
        };

        [shader("vertex")]
        VertexOutput vs_main(VertexInput input) {
            VertexOutput output;
            output.position = float4(input.position, 0.0, 1.0);
            output.color = input.color;
            return output;
        }

        [shader("fragment")]
        float4 fs_main(VertexOutput input) : SV_Target {
            return input.color;
        }
    "#;

    let shader = ShaderModule::from_slang(device, shader_source).expect("Failed to create shader");

    let pipeline = RenderPipeline::new(
        device,
        &shader,
        &shader,
        &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        },
    )
    .expect("Failed to create pipeline");

    let mut pool = goldy::RetainedPool::new(Arc::new(device.clone()));
    let vertex_buffer = pool
        .acquire_buffer_with_data(&vertices, BufferKind::Scattered)
        .expect("vertex buffer");
    let readback = acquire_readback_texture(&mut pool, width, height, TextureFormat::Rgba8Unorm);

    scheme_render_and_readback(&ctx, &target, &readback, "triangle", |pass| {
        pass.bind_parcel_mut(&vertex_buffer, NodeAccess::Read);
        pass.clear(clear_color);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertex_buffer);
        pass.draw(0..3, 0..1);
    })
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Depth3DVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}
impl goldy::StructuredBufferElement for Depth3DVertex {}

pub fn depth_vertex_layout() -> VertexBufferLayout {
    VertexBufferLayout {
        stride: std::mem::size_of::<Depth3DVertex>() as u32,
        attributes: vec![
            VertexAttribute {
                location: 0,
                format: VertexFormat::Float32x3,
                offset: 0,
            },
            VertexAttribute {
                location: 1,
                format: VertexFormat::Float32x4,
                offset: 12,
            },
        ],
    }
}

pub fn scheme_render_depth_occlusion(device: &Device, width: u32, height: u32) -> Vec<u8> {
    let ctx = device.create_context().expect("context");
    let target = RenderTarget::new_with_depth(
        device,
        width,
        height,
        TextureFormat::Rgba8Unorm,
        Some(DepthFormat::Depth32Float),
    )
    .expect("Failed to create render target with depth");

    let shader_source = include_str!("../../shaders/depth_test.slang");
    let shader = ShaderModule::from_slang(device, shader_source).expect("Failed to create shader");

    let pipeline = RenderPipeline::new(
        device,
        &shader,
        &shader,
        &RenderPipelineDesc {
            vertex_layout: depth_vertex_layout(),
            target_format: TextureFormat::Rgba8Unorm,
            depth_stencil: Some(DepthStencilState {
                format: DepthFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
            }),
            ..Default::default()
        },
    )
    .expect("Failed to create depth pipeline");

    let make_tri = |z: f32, color: [f32; 4]| -> [Depth3DVertex; 3] {
        [
            Depth3DVertex {
                position: [-1.0, -1.0, z],
                color,
            },
            Depth3DVertex {
                position: [3.0, -1.0, z],
                color,
            },
            Depth3DVertex {
                position: [-1.0, 3.0, z],
                color,
            },
        ]
    };

    let red_verts = make_tri(0.2, [1.0, 0.0, 0.0, 1.0]);
    let green_verts = make_tri(0.6, [0.0, 1.0, 0.0, 1.0]);

    let mut pool = goldy::RetainedPool::new(Arc::new(device.clone()));
    let red_vb = pool
        .acquire_buffer_with_data(&red_verts, BufferKind::Scattered)
        .expect("red vb");
    let green_vb = pool
        .acquire_buffer_with_data(&green_verts, BufferKind::Scattered)
        .expect("green vb");
    let readback = acquire_readback_texture(&mut pool, width, height, TextureFormat::Rgba8Unorm);

    scheme_render_and_readback(&ctx, &target, &readback, "depth_occlusion", |pass| {
        pass.bind_parcel_mut(&red_vb, NodeAccess::Read);
        pass.bind_parcel_mut(&green_vb, NodeAccess::Read);
        pass.clear(Color::BLACK);
        pass.clear_depth(1.0);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &red_vb);
        pass.draw(0..3, 0..1);
        pass.set_vertex_buffer(0, &green_vb);
        pass.draw(0..3, 0..1);
    })
}

pub fn scheme_render_game_of_life(device: &Device, updates: u32) -> Vec<u8> {
    let ctx = device.create_context().expect("context");
    const RENDER_WIDTH: u32 = 512;
    const RENDER_HEIGHT: u32 = 512;

    let compute_shader = ShaderModule::from_slang(device, include_str!("../../shaders/game_of_life.slang"))
        .expect("Failed to load compute shader");
    let render_shader = ShaderModule::from_slang(device, include_str!("../../shaders/game_of_life_render.slang"))
        .expect("Failed to load render shader");

    let initial_state = create_gol_initial_state();
    let mut pool = goldy::RetainedPool::new(Arc::new(device.clone()));
    let buffer_a = pool
        .acquire_buffer_with_data(&initial_state, BufferKind::Scattered)
        .expect("buffer_a");
    let buffer_b = pool
        .acquire_buffer_with_data(&initial_state, BufferKind::Scattered)
        .expect("buffer_b");

    let compute_pipeline = ComputePipeline::new(device, &compute_shader).expect("Failed to create compute pipeline");
    let render_pipeline = RenderPipeline::new(
        device,
        &render_shader,
        &render_shader,
        &RenderPipelineDesc {
            vertex_layout: VertexBufferLayout::default(),
            topology: PrimitiveTopology::TriangleList,
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        },
    )
    .expect("Failed to create render pipeline");

    let workgroups_x = GOL_GRID_WIDTH.div_ceil(8);
    let workgroups_y = GOL_GRID_HEIGHT.div_ceil(8);
    let mut use_buffer_a = true;

    for _ in 0..updates {
        let mut scheme = Scheme::new(&ctx);
        if use_buffer_a {
            let read = buffer_a.handle(ResourceAccess::ReadWrite).expect("buffer_a read");
            let write = buffer_b.handle(ResourceAccess::Write).expect("buffer_b write");
            scheme
                .node("gol_update", &compute_pipeline)
                .bind_parcel(&buffer_a, NodeAccess::Read)
                .bind_parcel(&buffer_b, NodeAccess::Write)
                .bind_views(&[read, write])
                .dispatch(workgroups_x, workgroups_y, 1);
        } else {
            let read = buffer_b.handle(ResourceAccess::ReadWrite).expect("buffer_b read");
            let write = buffer_a.handle(ResourceAccess::Write).expect("buffer_a write");
            scheme
                .node("gol_update", &compute_pipeline)
                .bind_parcel(&buffer_b, NodeAccess::Read)
                .bind_parcel(&buffer_a, NodeAccess::Write)
                .bind_views(&[read, write])
                .dispatch(workgroups_x, workgroups_y, 1);
        }
        scheme.submit().expect("compute submit");
        use_buffer_a = !use_buffer_a;
    }

    let target = RenderTarget::new(device, RENDER_WIDTH, RENDER_HEIGHT, TextureFormat::Rgba8Unorm)
        .expect("Failed to create render target");
    let readback = acquire_readback_texture(&mut pool, RENDER_WIDTH, RENDER_HEIGHT, TextureFormat::Rgba8Unorm);

    scheme_render_and_readback(&ctx, &target, &readback, "gol_render", |pass| {
        let cells = if use_buffer_a { &buffer_a } else { &buffer_b };
        // Scattered<uint> maps to a UAV slot; match TaskGraph bind_resources (ReadWrite), not SRV.
        pass.bind_shader_resources(&[ShaderResourceSlot::Parcel {
            parcel: cells,
            access: NodeAccess::ReadWrite,
        }]);
        pass.clear(Color::BLACK);
        pass.set_pipeline(&render_pipeline);
        pass.draw(0..3, 0..1);
    })
}
