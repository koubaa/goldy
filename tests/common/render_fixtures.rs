//! Shared offscreen rendering helpers for FLIP screenshot tests and the `update-screenshots` tool.
//!
//! Included from multiple integration test binaries; not every entry point is used in each crate.
#![allow(dead_code)]

use goldy::{
    BufferKind, Color, CompareFunction, ComputePipeline, DepthFormat, DepthStencilState, Device, DeviceDescriptor,
    Instance, NodeAccess, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RenderTarget, RequestAdapterOptions,
    ShaderModule, TaskGraph, TextureFormat, Vertex2D, VertexAttribute, VertexBufferLayout, VertexFormat,
};

fn test_alloc_buffer_with_data<T: goldy::StructuredBufferElement>(
    device: &goldy::Device,
    data: &[T],
    kind: goldy::BufferKind,
) -> goldy::Buffer {
    use std::sync::Arc;
    goldy::RetainedPool::new(Arc::new(device.clone()))
        .acquire_buffer_with_data(data, kind)
        .expect("acquire_buffer_with_data")
        .detach_buffer()
        .expect("detach_buffer")
}

fn graph_render(
    device: &Device,
    target: &RenderTarget,
    label: &'static str,
    record: impl FnOnce(&mut goldy::RenderPassBuilder<'_>),
) {
    let ctx = device.create_context().expect("context");
    let mut graph = TaskGraph::new();
    let mut pass = graph.render_pass(label, target);
    record(&mut pass);
    pass.finish_recorded();
    graph.dispatch(&ctx).expect("graph dispatch");
}

pub fn create_device() -> Option<Device> {
    let instance = Instance::new().ok()?;
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .ok()?
        .request_device(&DeviceDescriptor::default())
        .ok()
}

pub fn render_clear(device: &Device, width: u32, height: u32, color: Color) -> Vec<u8> {
    let target =
        RenderTarget::new(device, width, height, TextureFormat::Rgba8Unorm).expect("Failed to create render target");

    graph_render(device, &target, "clear", |pass| {
        pass.clear(color);
    });
    target.read_to_cpu().expect("Failed to read pixels")
}

pub fn render_triangle(
    device: &Device,
    width: u32,
    height: u32,
    clear_color: Color,
    vertices: [Vertex2D; 3],
) -> Vec<u8> {
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

    let vertex_buffer = test_alloc_buffer_with_data(&device, &vertices, BufferKind::Scattered);

    graph_render(device, &target, "triangle", |pass| {
        pass.bind_buffer_mut(&vertex_buffer, NodeAccess::Read);
        pass.clear(clear_color);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertex_buffer);
        pass.draw(0..3, 0..1);
    });
    target.read_to_cpu().expect("Failed to read pixels")
}

// ============================================================================
// Game of Life
// ============================================================================

pub const GOL_GRID_WIDTH: u32 = 128;
pub const GOL_GRID_HEIGHT: u32 = 128;
pub const GOL_CELL_COUNT: u32 = GOL_GRID_WIDTH * GOL_GRID_HEIGHT;

pub fn create_gol_initial_state() -> Vec<u32> {
    let mut cells = vec![0u32; GOL_CELL_COUNT as usize];

    let gun = [
        (1, 5),
        (1, 6),
        (2, 5),
        (2, 6),
        (11, 5),
        (11, 6),
        (11, 7),
        (12, 4),
        (12, 8),
        (13, 3),
        (13, 9),
        (14, 3),
        (14, 9),
        (15, 6),
        (16, 4),
        (16, 8),
        (17, 5),
        (17, 6),
        (17, 7),
        (18, 6),
        (21, 3),
        (21, 4),
        (21, 5),
        (22, 3),
        (22, 4),
        (22, 5),
        (23, 2),
        (23, 6),
        (25, 1),
        (25, 2),
        (25, 6),
        (25, 7),
        (35, 3),
        (35, 4),
        (36, 3),
        (36, 4),
    ];

    let offset_x = 10;
    let offset_y = 10;
    for (x, y) in gun.iter() {
        let px = (x + offset_x) as u32;
        let py = (y + offset_y) as u32;
        if px < GOL_GRID_WIDTH && py < GOL_GRID_HEIGHT {
            cells[(py * GOL_GRID_WIDTH + px) as usize] = 1;
        }
    }

    let seed = 42u64;
    let mut rng = seed;
    for y in 60..100 {
        for x in 60..100 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32).is_multiple_of(4) {
                cells[(y * GOL_GRID_WIDTH + x) as usize] = 1;
            }
        }
    }

    cells
}

pub fn render_game_of_life(device: &Device, updates: u32) -> Vec<u8> {
    let ctx = device.create_context().expect("context");
    let render_width = 512u32;
    let render_height = 512u32;

    let compute_shader = ShaderModule::from_slang(device, include_str!("../../shaders/game_of_life.slang"))
        .expect("Failed to load compute shader");

    let render_shader = ShaderModule::from_slang(device, include_str!("../../shaders/game_of_life_render.slang"))
        .expect("Failed to load render shader");

    let initial_state = create_gol_initial_state();
    let buffer_a = test_alloc_buffer_with_data(&device, &initial_state, BufferKind::Scattered);
    let buffer_b = test_alloc_buffer_with_data(&device, &initial_state, BufferKind::Scattered);

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

    let mut use_buffer_a = true;
    let workgroups_x = GOL_GRID_WIDTH.div_ceil(8);
    let workgroups_y = GOL_GRID_HEIGHT.div_ceil(8);

    for _ in 0..updates {
        let mut graph = TaskGraph::new();
        if use_buffer_a {
            graph
                .node("gol_update", &compute_pipeline)
                .bind_resources(&[&buffer_a, &buffer_b])
                .dispatch(workgroups_x, workgroups_y, 1);
        } else {
            graph
                .node("gol_update", &compute_pipeline)
                .bind_resources(&[&buffer_b, &buffer_a])
                .dispatch(workgroups_x, workgroups_y, 1);
        }
        graph.dispatch(&ctx).expect("Compute dispatch failed");
        use_buffer_a = !use_buffer_a;
    }

    let target = RenderTarget::new(device, render_width, render_height, TextureFormat::Rgba8Unorm)
        .expect("Failed to create render target");

    graph_render(device, &target, "gol_render", |pass| {
        if use_buffer_a {
            pass.bind_buffer_mut(&buffer_a, NodeAccess::Read);
            pass.clear(Color::BLACK);
            pass.set_pipeline(&render_pipeline);
            pass.bind_resources(&[&buffer_a]);
        } else {
            pass.bind_buffer_mut(&buffer_b, NodeAccess::Read);
            pass.clear(Color::BLACK);
            pass.set_pipeline(&render_pipeline);
            pass.bind_resources(&[&buffer_b]);
        }
        pass.draw(0..3, 0..1);
    });
    target.read_to_cpu().expect("Failed to read pixels")
}

// ============================================================================
// Depth occlusion
// ============================================================================

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

pub fn render_depth_occlusion(device: &Device, width: u32, height: u32) -> Vec<u8> {
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

    let red_vb = test_alloc_buffer_with_data(&device, &red_verts, BufferKind::Scattered);
    let green_vb = test_alloc_buffer_with_data(&device, &green_verts, BufferKind::Scattered);

    graph_render(device, &target, "depth_occlusion", |pass| {
        pass.bind_buffer_mut(&red_vb, NodeAccess::Read);
        pass.bind_buffer_mut(&green_vb, NodeAccess::Read);
        pass.clear(Color::BLACK);
        pass.clear_depth(1.0);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &red_vb);
        pass.draw(0..3, 0..1);
        pass.set_vertex_buffer(0, &green_vb);
        pass.draw(0..3, 0..1);
    });
    target.read_to_cpu().expect("Failed to read pixels")
}
