//! Screenshot tests for Goldy examples using FLIP perceptual image comparison.
//!
//! These tests render examples to offscreen targets and compare them against
//! reference PNG images using NVIDIA's FLIP algorithm.
//!
//! ## Running Tests
//!
//! ```bash
//! cargo test --test screenshot_tests
//! ```
//!
//! ## Creating Reference Images
//!
//! Reference images must be created manually and placed in `tests/screenshots/`.
//! See the README in that directory for details.

mod common;

use std::path::Path;

use common::image::{compare_images, ComparisonType, ImageComparisonError};
use goldy::{
    Buffer, Color, CommandEncoder, CompareFunction, ComputeEncoder, ComputePipeline, DataAccess,
    DepthFormat, DepthStencilState, Device, DeviceType, Instance, PrimitiveTopology,
    RenderPipeline, RenderPipelineDesc, RenderTarget, ShaderModule, TextureFormat, Vertex2D,
    VertexAttribute, VertexBufferLayout, VertexFormat,
};

fn create_device() -> Option<Device> {
    let instance = Instance::new().ok()?;
    instance.create_device(DeviceType::DiscreteGpu).ok()
}

fn run_screenshot_test(
    name: &str,
    reference_path: &str,
    width: u32,
    height: u32,
    comparisons: &[ComparisonType],
    pixels: Vec<u8>,
) {
    println!("Running screenshot test: {}", name);

    let reference_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(reference_path);

    match compare_images(&reference_path, width, height, &pixels, comparisons) {
        Ok(()) => {
            println!("Screenshot test '{}' passed!", name);
        }
        Err(ImageComparisonError::ReferenceNotFound(path)) => {
            panic!(
                "Reference image not found: {}\n\
                 To create it, run: cargo test --test generate_screenshots",
                path
            );
        }
        Err(e) => {
            panic!("Screenshot test '{}' failed: {}", name, e);
        }
    }
}

fn render_clear(device: &Device, width: u32, height: u32, color: Color) -> Vec<u8> {
    let target = RenderTarget::new(device, width, height, TextureFormat::Rgba8Unorm)
        .expect("Failed to create render target");

    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(color);
    }

    target.render(encoder).expect("Failed to render");
    target.read_to_cpu().expect("Failed to read pixels")
}

fn render_triangle(
    device: &Device,
    width: u32,
    height: u32,
    clear_color: Color,
    vertices: [Vertex2D; 3],
) -> Vec<u8> {
    let target = RenderTarget::new(device, width, height, TextureFormat::Rgba8Unorm)
        .expect("Failed to create render target");

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

    let vertex_buffer =
        Buffer::with_data(device, &vertices, DataAccess::Scattered).expect("Failed to create VB");

    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(clear_color);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertex_buffer);
        pass.draw(0..3, 0..1);
    }

    target.render(encoder).expect("Failed to render");
    target.read_to_cpu().expect("Failed to read pixels")
}

/// Test rendering a solid red color.
#[test]
fn test_solid_red() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let pixels = render_clear(&device, 64, 64, Color::RED);
    run_screenshot_test(
        "solid_red",
        "tests/screenshots/solid_red.png",
        64,
        64,
        &[ComparisonType::Mean(0.001)],
        pixels,
    );
}

/// Test rendering a solid blue color.
#[test]
fn test_solid_blue() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let pixels = render_clear(&device, 64, 64, Color::BLUE);
    run_screenshot_test(
        "solid_blue",
        "tests/screenshots/solid_blue.png",
        64,
        64,
        &[ComparisonType::Mean(0.001)],
        pixels,
    );
}

/// Test rendering the classic RGB triangle.
#[test]
fn test_rgb_triangle() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let vertices = [
        Vertex2D::new(0.0, -0.8, Color::RED),
        Vertex2D::new(-0.8, 0.8, Color::GREEN),
        Vertex2D::new(0.8, 0.8, Color::BLUE),
    ];

    let pixels = render_triangle(&device, 256, 256, Color::BLACK, vertices);
    run_screenshot_test(
        "rgb_triangle",
        "tests/screenshots/rgb_triangle.png",
        256,
        256,
        &[ComparisonType::Mean(0.02)],
        pixels,
    );
}

/// Test rendering a white triangle on black background.
#[test]
fn test_white_triangle() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let vertices = [
        Vertex2D::new(0.0, -0.5, Color::WHITE),
        Vertex2D::new(-0.5, 0.5, Color::WHITE),
        Vertex2D::new(0.5, 0.5, Color::WHITE),
    ];

    let pixels = render_triangle(&device, 128, 128, Color::BLACK, vertices);
    run_screenshot_test(
        "white_triangle",
        "tests/screenshots/white_triangle.png",
        128,
        128,
        &[ComparisonType::Mean(0.01)],
        pixels,
    );
}

// ============================================================================
// Game of Life Tests
// ============================================================================

const GOL_GRID_WIDTH: u32 = 128;
const GOL_GRID_HEIGHT: u32 = 128;
const GOL_CELL_COUNT: u32 = GOL_GRID_WIDTH * GOL_GRID_HEIGHT;

/// Create initial Game of Life state (Gosper Glider Gun + random cells)
fn create_gol_initial_state() -> Vec<u32> {
    let mut cells = vec![0u32; GOL_CELL_COUNT as usize];

    // Gosper Glider Gun
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

    // Add random cells in lower right
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

/// Run Game of Life simulation for the specified number of updates and render the result.
fn render_game_of_life(device: &Device, updates: u32) -> Vec<u8> {
    // Use 512x512 for better visual detail
    let render_width = 512u32;
    let render_height = 512u32;

    // Load shaders
    let compute_shader =
        ShaderModule::from_slang(device, include_str!("../shaders/game_of_life.slang"))
            .expect("Failed to load compute shader");

    let render_shader =
        ShaderModule::from_slang(device, include_str!("../shaders/game_of_life_render.slang"))
            .expect("Failed to load render shader");

    // Create ping-pong buffers
    let initial_state = create_gol_initial_state();
    let buffer_a = Buffer::with_data(device, &initial_state, DataAccess::Scattered)
        .expect("Failed to create buffer A");
    let buffer_b = Buffer::with_data(device, &initial_state, DataAccess::Scattered)
        .expect("Failed to create buffer B");

    // Create pipelines
    let compute_pipeline =
        ComputePipeline::new(device, &compute_shader).expect("Failed to create compute pipeline");

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

    // Run simulation
    let mut use_buffer_a = true;
    let workgroups_x = GOL_GRID_WIDTH.div_ceil(8);
    let workgroups_y = GOL_GRID_HEIGHT.div_ceil(8);

    for _ in 0..updates {
        let mut compute_encoder = ComputeEncoder::new();
        {
            let mut pass = compute_encoder.begin_compute_pass();
            pass.set_pipeline(&compute_pipeline);
            // Bindless: pass buffer indices via push constants
            if use_buffer_a {
                pass.set_push_constants(&[&buffer_a, &buffer_b]);
            } else {
                pass.set_push_constants(&[&buffer_b, &buffer_a]);
            }
            pass.dispatch(workgroups_x, workgroups_y, 1);
        }
        compute_encoder
            .dispatch(device)
            .expect("Compute dispatch failed");
        use_buffer_a = !use_buffer_a;
    }

    // Render
    let target = RenderTarget::new(
        device,
        render_width,
        render_height,
        TextureFormat::Rgba8Unorm,
    )
    .expect("Failed to create render target");

    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(Color::BLACK);
        pass.set_pipeline(&render_pipeline);
        // Bindless: pass buffer index via push constants
        if use_buffer_a {
            pass.set_push_constants(&[&buffer_a]);
        } else {
            pass.set_push_constants(&[&buffer_b]);
        }
        pass.draw(0..3, 0..1);
    }

    target.render(encoder).expect("Failed to render");
    target.read_to_cpu().expect("Failed to read pixels")
}

/// Test Game of Life at update 50.
#[test]
fn test_game_of_life_update_50() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let pixels = render_game_of_life(&device, 50);
    run_screenshot_test(
        "game_of_life_50",
        "tests/screenshots/game_of_life_50.png",
        512,
        512,
        &[ComparisonType::Mean(0.01)],
        pixels,
    );
}

/// Test Game of Life at update 100.
#[test]
fn test_game_of_life_update_100() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let pixels = render_game_of_life(&device, 100);
    run_screenshot_test(
        "game_of_life_100",
        "tests/screenshots/game_of_life_100.png",
        512,
        512,
        &[ComparisonType::Mean(0.01)],
        pixels,
    );
}

// ============================================================================
// Depth Occlusion Tests
// ============================================================================

/// Vertex type carrying (x, y, z) position and RGBA color for depth tests.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Depth3DVertex {
    position: [f32; 3],
    color: [f32; 4],
}

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

/// Render two overlapping fullscreen triangles at different depths.
///
/// Red triangle is drawn first at z=0.2 (near).
/// Green triangle is drawn second at z=0.6 (far).
///
/// With depth testing (`Less`): the green triangle is occluded by red.
/// Without depth testing: green would overwrite red (since it's drawn second).
///
/// Expected output: solid red.
fn render_depth_occlusion(device: &Device, width: u32, height: u32) -> Vec<u8> {
    let target = RenderTarget::new_with_depth(
        device,
        width,
        height,
        TextureFormat::Rgba8Unorm,
        Some(DepthFormat::Depth32Float),
    )
    .expect("Failed to create render target with depth");

    let shader_source = include_str!("../shaders/depth_test.slang");
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

    // Fullscreen triangle (covers entire NDC [-1,1]²) using the large-triangle trick.
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

    // Red at z=0.2 (front), green at z=0.6 (back)
    let red_verts = make_tri(0.2, [1.0, 0.0, 0.0, 1.0]);
    let green_verts = make_tri(0.6, [0.0, 1.0, 0.0, 1.0]);

    let red_vb =
        Buffer::with_data(device, &red_verts, DataAccess::Scattered).expect("Failed to create VB");
    let green_vb = Buffer::with_data(device, &green_verts, DataAccess::Scattered)
        .expect("Failed to create VB");

    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(Color::BLACK);
        pass.clear_depth(1.0);
        pass.set_pipeline(&pipeline);

        // Draw red first (z=0.2, writes depth)
        pass.set_vertex_buffer(0, &red_vb);
        pass.draw(0..3, 0..1);

        // Draw green second (z=0.6, fails depth test: 0.6 is NOT Less than 0.2)
        pass.set_vertex_buffer(0, &green_vb);
        pass.draw(0..3, 0..1);
    }

    target.render(encoder).expect("Failed to render");
    target.read_to_cpu().expect("Failed to read pixels")
}

/// Depth occlusion test: a near (red) geometry blocks a far (green) geometry.
///
/// Verifies that the depth buffer correctly occludes back-facing geometry
/// even when it is submitted to the GPU after the front geometry.
#[test]
fn test_depth_occlusion() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let pixels = render_depth_occlusion(&device, 64, 64);
    run_screenshot_test(
        "depth_occlusion",
        "tests/screenshots/depth_occlusion.png",
        64,
        64,
        &[ComparisonType::Mean(0.001)],
        pixels,
    );
}
