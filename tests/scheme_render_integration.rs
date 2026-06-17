//! Scheme render integration tests — duplicated from TaskGraph coverage.
//!
//! Original tests in `render_target_integration.rs` and `surface_graph_integration.rs`
//! remain until ekrano migration completes.
#![cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]

#[path = "common/scheme_render.rs"]
mod scheme_render;
#[path = "common/submission.rs"]
mod submission;

use goldy::{
    shader::builtins,
    types::{AddressMode, FilterMode, SamplerDesc, TextureFlags, TextureKind},
    BufferKind, Color, CompareFunction, DepthFormat, DepthStencilState, IndexFormat, NodeAccess, PrimitiveTopology,
    RenderPipeline, RenderPipelineDesc, Sampler, ShaderModule, ShaderResourceSlot, TextureFormat, Vertex2D, Vertex2DUv,
    VertexAttribute, VertexBufferLayout, VertexFormat,
};
use scheme_render::{
    acquire_readback_texture, device_and_pool, read_grant_texture, scheme_record_readback, scheme_render_and_readback,
};
use submission::submission_context;

#[test]
fn scheme_render_pass_triangle_readback() {
    let Some((device, mut pool)) = device_and_pool() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };
    let ctx = submission_context(&device);

    const W: u32 = 64;
    const H: u32 = 64;
    let clear = Color {
        r: 0.1,
        g: 0.1,
        b: 0.2,
        a: 1.0,
    };
    let vertices = [
        Vertex2D::new(0.0, -0.5, Color::RED),
        Vertex2D::new(-0.5, 0.5, Color::GREEN),
        Vertex2D::new(0.5, 0.5, Color::BLUE),
    ];
    let vertex_buffer = pool
        .acquire_buffer_with_data(&vertices, BufferKind::Scattered)
        .expect("vertex buffer");

    let readback = acquire_readback_texture(&mut pool, W, H, TextureFormat::Rgba8Unorm);
    let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D).expect("shader");
    let pipeline = RenderPipeline::new(
        &device,
        &shader,
        &shader,
        &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        },
    )
    .expect("pipeline");

    let pixels = scheme_render_and_readback(
        &ctx,
        W,
        H,
        TextureFormat::Rgba8Unorm,
        None,
        &readback,
        "triangle",
        |pass| {
            pass.bind_parcel_mut(&vertex_buffer, NodeAccess::Read);
            pass.clear(clear);
            pass.set_pipeline(&pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.draw(0..3, 0..1);
        },
    );

    let stride = (W * 4) as usize;
    let cx = (W / 2) as usize;
    let cy = (H / 2) as usize;
    let i = cy * stride + cx * 4;
    let r = pixels[i];
    let g = pixels[i + 1];
    let b = pixels[i + 2];

    assert!(
        r > 20 || g > 20 || b > 20,
        "center pixel should be lit by the triangle, got rgba=({r},{g},{b},{})",
        pixels[i + 3]
    );
    assert!(
        (r as i32 - (clear.r * 255.0) as i32).abs() > 5
            || (g as i32 - (clear.g * 255.0) as i32).abs() > 5
            || (b as i32 - (clear.b * 255.0) as i32).abs() > 5,
        "center pixel should differ from clear color"
    );
}

#[test]
fn scheme_vulkan_render_and_readback() {
    let Some((device, mut pool)) = device_and_pool() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };
    let ctx = submission_context(&device);

    let shader_source = r#"
        struct VertexInput {
            float2 position : POSITION;
            float4 color : COLOR;
        };

        struct VertexOutput {
            float4 position : SV_Position;
            float4 color : COLOR;
        };

        [goldy_vertex]
        VertexOutput vs_main(VertexInput input) {
            VertexOutput output;
            output.position = float4(input.position, 0.0, 1.0);
            output.color = input.color;
            return output;
        }

        [goldy_fragment]
        float4 fs_main(VertexOutput input) : SV_Target {
            return input.color;
        }
    "#;

    let shader = ShaderModule::from_slang(&device, shader_source).expect("shader");
    let pipeline = RenderPipeline::new(
        &device,
        &shader,
        &shader,
        &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        },
    )
    .expect("pipeline");

    let vertices = [
        Vertex2D::new(0.0, -0.5, Color::RED),
        Vertex2D::new(-0.5, 0.5, Color::GREEN),
        Vertex2D::new(0.5, 0.5, Color::BLUE),
    ];
    let vertex_buffer = pool
        .acquire_buffer_with_data(&vertices, BufferKind::Scattered)
        .expect("vertex buffer");
    let readback = acquire_readback_texture(&mut pool, 100, 100, TextureFormat::Rgba8Unorm);

    let pixels = scheme_render_and_readback(
        &ctx,
        100,
        100,
        TextureFormat::Rgba8Unorm,
        None,
        &readback,
        "triangle",
        |pass| {
            pass.bind_parcel_mut(&vertex_buffer, NodeAccess::Read);
            pass.clear(Color::BLACK);
            pass.set_pipeline(&pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.draw(0..3, 0..1);
        },
    );

    assert_eq!(pixels.len(), 100 * 100 * 4);
    let has_non_black = pixels.chunks(4).any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0);
    assert!(has_non_black, "Expected rendered triangle to have non-black pixels");
}

#[test]
fn scheme_render_target_clear_only() {
    let Some((device, mut pool)) = device_and_pool() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };
    let ctx = submission_context(&device);

    let readback = acquire_readback_texture(&mut pool, 4, 4, TextureFormat::Rgba8Unorm);
    let clear_color = Color::from_rgb(128, 64, 32);

    let pixels = scheme_render_and_readback(
        &ctx,
        4,
        4,
        TextureFormat::Rgba8Unorm,
        None,
        &readback,
        "clear",
        |pass| {
            pass.clear(clear_color);
        },
    );

    for chunk in pixels.chunks(4) {
        assert_eq!(chunk[0], 128, "Red channel mismatch");
        assert_eq!(chunk[1], 64, "Green channel mismatch");
        assert_eq!(chunk[2], 32, "Blue channel mismatch");
        assert_eq!(chunk[3], 255, "Alpha channel mismatch");
    }

    // Retention smoke: scheme records once, second submit resubmits.
    let readback2 = acquire_readback_texture(&mut pool, 4, 4, TextureFormat::Rgba8Unorm);
    let (mut scheme, _grant) = scheme_record_readback(
        &ctx,
        4,
        4,
        TextureFormat::Rgba8Unorm,
        None,
        &readback2,
        "clear",
        |pass| {
            pass.clear(clear_color);
        },
    );
    let frame0 = scheme.submit().expect("submit 0");
    assert_eq!(scheme.replay_stats().records, 1);
    let frame1 = scheme.submit().expect("submit 1");
    #[cfg(not(feature = "metal"))]
    assert_eq!(scheme.replay_stats().resubmit_hits, 1);
    drop(frame0);
    drop(frame1);
}

#[test]
fn scheme_steady_state_readback_loop() {
    let Some((device, mut pool)) = device_and_pool() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };
    let ctx = submission_context(&device);

    let readback = acquire_readback_texture(&mut pool, 4, 4, TextureFormat::Rgba8Unorm);
    let clear_color = Color::from_rgb(128, 64, 32);

    let (mut scheme, grant) = scheme_record_readback(
        &ctx,
        4,
        4,
        TextureFormat::Rgba8Unorm,
        None,
        &readback,
        "clear",
        |pass| {
            pass.clear(clear_color);
        },
    );

    let frame0 = scheme.submit().expect("submit 0");
    let pixels0 = read_grant_texture(&grant, &frame0);
    let frame1 = scheme.submit().expect("submit 1");
    let pixels1 = read_grant_texture(&grant, &frame1);

    for chunk in pixels0.chunks(4) {
        assert_eq!(chunk[0], 128);
        assert_eq!(chunk[1], 64);
        assert_eq!(chunk[2], 32);
        assert_eq!(chunk[3], 255);
    }
    assert_eq!(
        pixels0, pixels1,
        "steady-state resubmit must produce identical readback"
    );
    assert_eq!(scheme.replay_stats().records, 1, "topology recorded once");
    #[cfg(not(feature = "metal"))]
    assert_eq!(
        scheme.replay_stats().resubmit_hits,
        1,
        "second submit resubmits retained render pass"
    );
}

#[test]
fn scheme_multiple_render_targets() {
    let Some((device, mut pool)) = device_and_pool() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };
    let ctx = submission_context(&device);

    let readback1 = acquire_readback_texture(&mut pool, 10, 10, TextureFormat::Rgba8Unorm);
    let readback2 = acquire_readback_texture(&mut pool, 20, 20, TextureFormat::Rgba8Unorm);

    let pixels1 = scheme_render_and_readback(
        &ctx,
        10,
        10,
        TextureFormat::Rgba8Unorm,
        None,
        &readback1,
        "clear_red",
        |pass| {
            pass.clear(Color::RED);
        },
    );
    let pixels2 = scheme_render_and_readback(
        &ctx,
        20,
        20,
        TextureFormat::Rgba8Unorm,
        None,
        &readback2,
        "clear_blue",
        |pass| {
            pass.clear(Color::BLUE);
        },
    );

    assert_eq!(pixels1[0], 255);
    assert_eq!(pixels1[1], 0);
    assert_eq!(pixels1[2], 0);

    assert_eq!(pixels2[0], 0);
    assert_eq!(pixels2[1], 0);
    assert_eq!(pixels2[2], 255);
}

#[test]
fn scheme_indexed_drawing() {
    let Some((device, mut pool)) = device_and_pool() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };
    let ctx = submission_context(&device);

    let shader_source = r#"
        struct VertexInput {
            float2 position : POSITION;
            float4 color : COLOR;
        };

        struct VertexOutput {
            float4 position : SV_Position;
            float4 color : COLOR;
        };

        [goldy_vertex]
        VertexOutput vs_main(VertexInput input) {
            VertexOutput output;
            output.position = float4(input.position, 0.0, 1.0);
            output.color = input.color;
            return output;
        }

        [goldy_fragment]
        float4 fs_main(VertexOutput input) : SV_Target {
            return input.color;
        }
    "#;

    let shader = ShaderModule::from_slang(&device, shader_source).expect("shader");
    let pipeline = RenderPipeline::new(
        &device,
        &shader,
        &shader,
        &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        },
    )
    .expect("pipeline");

    let vertices = [
        Vertex2D::new(-0.5, -0.5, Color::RED),
        Vertex2D::new(0.5, -0.5, Color::GREEN),
        Vertex2D::new(0.5, 0.5, Color::BLUE),
        Vertex2D::new(-0.5, 0.5, Color::WHITE),
    ];
    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
    let vertex_buffer = pool
        .acquire_buffer_with_data(&vertices, BufferKind::Scattered)
        .expect("vb");
    let index_buffer = pool
        .acquire_buffer_with_data(&indices, BufferKind::Scattered)
        .expect("ib");
    let readback = acquire_readback_texture(&mut pool, 100, 100, TextureFormat::Rgba8Unorm);

    let pixels = scheme_render_and_readback(
        &ctx,
        100,
        100,
        TextureFormat::Rgba8Unorm,
        None,
        &readback,
        "indexed_u16",
        |pass| {
            pass.bind_parcel_mut(&vertex_buffer, NodeAccess::Read);
            pass.bind_parcel_mut(&index_buffer, NodeAccess::Read);
            pass.clear(Color::BLACK);
            pass.set_pipeline(&pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.set_index_buffer(&index_buffer, IndexFormat::Uint16);
            pass.draw_indexed(0..6, 0, 0..1);
        },
    );

    assert_eq!(pixels.len(), 100 * 100 * 4);
    let non_black_count = pixels.chunks(4).filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count();
    assert!(
        non_black_count > 1000,
        "Expected quad to cover at least 1000 pixels, got {}",
        non_black_count
    );
}

#[test]
fn scheme_indexed_drawing_uint32() {
    let Some((device, mut pool)) = device_and_pool() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };
    let ctx = submission_context(&device);

    let shader_source = r#"
        struct VertexInput {
            float2 position : POSITION;
            float4 color : COLOR;
        };

        struct VertexOutput {
            float4 position : SV_Position;
            float4 color : COLOR;
        };

        [goldy_vertex]
        VertexOutput vs_main(VertexInput input) {
            VertexOutput output;
            output.position = float4(input.position, 0.0, 1.0);
            output.color = input.color;
            return output;
        }

        [goldy_fragment]
        float4 fs_main(VertexOutput input) : SV_Target {
            return input.color;
        }
    "#;

    let shader = ShaderModule::from_slang(&device, shader_source).expect("shader");
    let pipeline = RenderPipeline::new(
        &device,
        &shader,
        &shader,
        &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        },
    )
    .expect("pipeline");

    let vertices = [
        Vertex2D::new(0.0, -0.8, Color::RED),
        Vertex2D::new(-0.8, 0.8, Color::RED),
        Vertex2D::new(0.8, 0.8, Color::RED),
    ];
    let indices: [u32; 3] = [0, 1, 2];
    let vertex_buffer = pool
        .acquire_buffer_with_data(&vertices, BufferKind::Scattered)
        .expect("vb");
    let index_buffer = pool
        .acquire_buffer_with_data(&indices, BufferKind::Scattered)
        .expect("ib");
    let readback = acquire_readback_texture(&mut pool, 50, 50, TextureFormat::Rgba8Unorm);

    let pixels = scheme_render_and_readback(
        &ctx,
        50,
        50,
        TextureFormat::Rgba8Unorm,
        None,
        &readback,
        "indexed_u32",
        |pass| {
            pass.bind_parcel_mut(&vertex_buffer, NodeAccess::Read);
            pass.bind_parcel_mut(&index_buffer, NodeAccess::Read);
            pass.clear(Color::BLACK);
            pass.set_pipeline(&pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.set_index_buffer(&index_buffer, IndexFormat::Uint32);
            pass.draw_indexed(0..3, 0, 0..1);
        },
    );

    let red_pixel_count = pixels
        .chunks(4)
        .filter(|p| p[0] > 200 && p[1] < 50 && p[2] < 50)
        .count();
    assert!(
        red_pixel_count > 100,
        "Expected red triangle pixels, got {}",
        red_pixel_count
    );
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Depth3DVertex {
    position: [f32; 3],
    color: [f32; 4],
}
impl goldy::StructuredBufferElement for Depth3DVertex {}

fn depth_vertex_layout() -> VertexBufferLayout {
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

#[test]
fn scheme_depth_occlusion_red_beats_green() {
    let Some((device, mut pool)) = device_and_pool() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };
    let ctx = submission_context(&device);

    let shader = ShaderModule::from_slang(&device, include_str!("../shaders/depth_test.slang")).expect("shader");
    let pipeline = RenderPipeline::new(
        &device,
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
    .expect("pipeline");

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
    let red_vb = pool
        .acquire_buffer_with_data(&red_verts, BufferKind::Scattered)
        .expect("red vb");
    let green_vb = pool
        .acquire_buffer_with_data(&green_verts, BufferKind::Scattered)
        .expect("green vb");
    let readback = acquire_readback_texture(&mut pool, 64, 64, TextureFormat::Rgba8Unorm);

    let pixels = scheme_render_and_readback(
        &ctx,
        64,
        64,
        TextureFormat::Rgba8Unorm,
        Some(DepthFormat::Depth32Float),
        &readback,
        "depth_red_wins",
        |pass| {
            pass.bind_parcel_mut(&red_vb, NodeAccess::Read);
            pass.bind_parcel_mut(&green_vb, NodeAccess::Read);
            pass.clear(Color::BLACK);
            pass.clear_depth(1.0);
            pass.set_pipeline(&pipeline);
            pass.set_vertex_buffer(0, &red_vb);
            pass.draw(0..3, 0..1);
            pass.set_vertex_buffer(0, &green_vb);
            pass.draw(0..3, 0..1);
        },
    );

    let total = pixels.len() / 4;
    let red_count = pixels
        .chunks(4)
        .filter(|p| p[0] > 200 && p[1] < 50 && p[2] < 50)
        .count();
    let green_count = pixels
        .chunks(4)
        .filter(|p| p[1] > 200 && p[0] < 50 && p[2] < 50)
        .count();

    assert!(
        red_count == total,
        "Depth occlusion failed: expected all {} pixels red, got {} red / {} green",
        total,
        red_count,
        green_count
    );
}

#[test]
fn scheme_depth_occlusion_green_beats_red() {
    let Some((device, mut pool)) = device_and_pool() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };
    let ctx = submission_context(&device);

    let shader = ShaderModule::from_slang(&device, include_str!("../shaders/depth_test.slang")).expect("shader");
    let pipeline = RenderPipeline::new(
        &device,
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
    .expect("pipeline");

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

    let red_verts = make_tri(0.8, [1.0, 0.0, 0.0, 1.0]);
    let green_verts = make_tri(0.2, [0.0, 1.0, 0.0, 1.0]);
    let red_vb = pool
        .acquire_buffer_with_data(&red_verts, BufferKind::Scattered)
        .expect("red vb");
    let green_vb = pool
        .acquire_buffer_with_data(&green_verts, BufferKind::Scattered)
        .expect("green vb");
    let readback = acquire_readback_texture(&mut pool, 64, 64, TextureFormat::Rgba8Unorm);

    let pixels = scheme_render_and_readback(
        &ctx,
        64,
        64,
        TextureFormat::Rgba8Unorm,
        Some(DepthFormat::Depth32Float),
        &readback,
        "depth_green_wins",
        |pass| {
            pass.bind_parcel_mut(&red_vb, NodeAccess::Read);
            pass.bind_parcel_mut(&green_vb, NodeAccess::Read);
            pass.clear(Color::BLACK);
            pass.clear_depth(1.0);
            pass.set_pipeline(&pipeline);
            pass.set_vertex_buffer(0, &red_vb);
            pass.draw(0..3, 0..1);
            pass.set_vertex_buffer(0, &green_vb);
            pass.draw(0..3, 0..1);
        },
    );

    let total = pixels.len() / 4;
    let green_count = pixels
        .chunks(4)
        .filter(|p| p[1] > 200 && p[0] < 50 && p[2] < 50)
        .count();
    let red_count = pixels
        .chunks(4)
        .filter(|p| p[0] > 200 && p[1] < 50 && p[2] < 50)
        .count();

    assert!(
        green_count == total,
        "Depth occlusion failed: expected all {} pixels green, got {} green / {} red",
        total,
        green_count,
        red_count
    );
}

#[test]
fn scheme_render_target_bindless_buffer_read() {
    let Some((device, mut pool)) = device_and_pool() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };
    let ctx = submission_context(&device);

    let data = vec![1u32; 4];
    let buffer = pool
        .acquire_buffer_with_data(&data, BufferKind::Scattered)
        .expect("buffer");

    let shader_source = r#"
import goldy_exp;

static const float2 positions[3] = {
    float2(-1, -1),
    float2( 3, -1),
    float2(-1,  3)
};

static const float2 uvs[3] = {
    float2(0, 0),
    float2(2, 0),
    float2(0, 2)
};

struct VSOut {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

[goldy_vertex]
VSOut vs_main(VertexId id) {
    VSOut o;
    o.pos = float4(positions[id.value], 0.0, 1.0);
    o.uv  = uvs[id.value];
    return o;
}

[goldy_fragment]
float4 fs_main(Scattered<uint> cells, VSOut i) : SV_Target {
    uint val = cells[0];
    if (val == 1u) {
        return float4(0.2, 0.9, 0.3, 1.0);
    } else {
        return float4(0.05, 0.08, 0.1, 1.0);
    }
}
"#;

    let shader = ShaderModule::from_slang(&device, shader_source).expect("shader");
    let readback = acquire_readback_texture(&mut pool, 4, 4, TextureFormat::Rgba8Unorm);

    let pipeline = RenderPipeline::new(
        &device,
        &shader,
        &shader,
        &RenderPipelineDesc {
            vertex_layout: VertexBufferLayout {
                attributes: vec![],
                stride: 0,
            },
            topology: PrimitiveTopology::TriangleList,
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        },
    )
    .expect("pipeline");

    let pixels = scheme_render_and_readback(
        &ctx,
        4,
        4,
        TextureFormat::Rgba8Unorm,
        None,
        &readback,
        "bindless_read",
        |pass| {
            pass.clear(Color::BLACK);
            pass.bind_shader_resources(&[ShaderResourceSlot::Parcel {
                parcel: &buffer,
                access: NodeAccess::ReadWrite,
            }]);
            pass.set_pipeline(&pipeline);
            pass.draw(0..3, 0..1);
        },
    );

    assert_eq!(pixels.len(), 4 * 4 * 4);
    let all_green = pixels.chunks(4).all(|p| p[1] > 100);
    assert!(
        all_green,
        "Expected all pixels green from bindless buffer read; first pixel: {:?}",
        &pixels[..4]
    );
}

#[test]
fn scheme_render_target_textured_quad_readback() {
    let Some((device, mut pool)) = device_and_pool() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };
    let ctx = submission_context(&device);

    const W: u32 = 64;
    const H: u32 = 64;
    const CHECKER: u32 = 8;

    let mut checker_data = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        for x in 0..W {
            let is_white = ((x / CHECKER) + (y / CHECKER)).is_multiple_of(2);
            if is_white {
                checker_data.extend_from_slice(&[255, 255, 255, 255]);
            } else {
                checker_data.extend_from_slice(&[50, 100, 200, 255]);
            }
        }
    }

    let texture = pool
        .acquire_texture(
            W,
            H,
            TextureFormat::Rgba8Unorm,
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
            Some(&checker_data),
        )
        .expect("texture");
    let sampler = Sampler::new(
        &device,
        &SamplerDesc {
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            max_anisotropy: 1.0,
            compare: None,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
        },
    )
    .expect("sampler");

    let vertices = [
        Vertex2DUv::new(-1.0, -1.0, 0.0, 1.0),
        Vertex2DUv::new(1.0, -1.0, 1.0, 1.0),
        Vertex2DUv::new(1.0, 1.0, 1.0, 0.0),
        Vertex2DUv::new(-1.0, -1.0, 0.0, 1.0),
        Vertex2DUv::new(1.0, 1.0, 1.0, 0.0),
        Vertex2DUv::new(-1.0, 1.0, 0.0, 0.0),
    ];
    let vertex_buffer = pool
        .acquire_buffer_with_data(&vertices, BufferKind::Scattered)
        .expect("vertex buffer");

    let shader_source = r#"
import goldy_exp;

[goldy_vertex]
FullscreenVarying vs_main(FullscreenVertex input) {
    return vs_fullscreen(input);
}

[goldy_fragment]
float4 fs_main(Interpolated<float4> tex, Filter smp, FullscreenVarying input) : SV_Target {
    return tex.Sample(smp, input.uv);
}
"#;

    let shader = ShaderModule::from_slang(&device, shader_source).expect("shader");
    let readback = acquire_readback_texture(&mut pool, W, H, TextureFormat::Rgba8Unorm);
    let pipeline = RenderPipeline::new(
        &device,
        &shader,
        &shader,
        &RenderPipelineDesc {
            vertex_layout: Vertex2DUv::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        },
    )
    .expect("pipeline");

    let pixels = scheme_render_and_readback(
        &ctx,
        W,
        H,
        TextureFormat::Rgba8Unorm,
        None,
        &readback,
        "textured_quad",
        |pass| {
            pass.bind_shader_resources(&[
                ShaderResourceSlot::Parcel {
                    parcel: &texture,
                    access: NodeAccess::Read,
                },
                ShaderResourceSlot::Sampler(&sampler),
            ]);
            pass.bind_parcel_mut(&vertex_buffer, NodeAccess::Read);
            pass.clear(Color {
                r: 0.1,
                g: 0.1,
                b: 0.15,
                a: 1.0,
            });
            pass.set_pipeline(&pipeline);
            pass.set_vertex_buffer(0, &vertex_buffer);
            pass.draw(0..6, 0..1);
        },
    );

    assert_eq!(pixels.len(), (W * H * 4) as usize);
    let center = ((H / 2) * W + (W / 2)) as usize * 4;
    let r = pixels[center];
    let g = pixels[center + 1];
    let b = pixels[center + 2];
    assert!(
        r > 20 || g > 20 || b > 20,
        "center pixel should show sampled checkerboard, got rgba=({r},{g},{b},{})",
        pixels[center + 3]
    );
}
