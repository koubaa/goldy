//! Utility to generate reference screenshots for FLIP tests.
//!
//! Run with: cargo test --test generate_screenshots -- --nocapture

mod common;

use goldy::{
    Buffer, Color, CommandEncoder, ComputeEncoder, ComputePipeline, DataAccess, Device,
    DeviceType, Instance, PrimitiveTopology, RenderPipeline, RenderPipelineDesc, RenderTarget,
    ShaderModule, TextureFormat, Vertex2D, VertexBufferLayout,
};
use std::path::Path;

fn create_device() -> Option<Device> {
    let instance = Instance::new().ok()?;
    instance.create_device(DeviceType::DiscreteGpu).ok()
}

fn save_png(path: &Path, width: u32, height: u32, rgba_data: &[u8]) {
    let img = image::RgbaImage::from_raw(width, height, rgba_data.to_vec())
        .expect("Failed to create image");
    img.save(path).expect("Failed to save PNG");
    println!("Saved: {}", path.display());
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

#[test]
fn generate_solid_red() {
    let Some(device) = create_device() else {
        eprintln!("Skipping: no GPU available");
        return;
    };

    let pixels = render_clear(&device, 64, 64, Color::RED);
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/screenshots/solid_red.png");
    save_png(&path, 64, 64, &pixels);
}

#[test]
fn generate_solid_blue() {
    let Some(device) = create_device() else {
        eprintln!("Skipping: no GPU available");
        return;
    };

    let pixels = render_clear(&device, 64, 64, Color::BLUE);
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/screenshots/solid_blue.png");
    save_png(&path, 64, 64, &pixels);
}

#[test]
fn generate_rgb_triangle() {
    let Some(device) = create_device() else {
        eprintln!("Skipping: no GPU available");
        return;
    };

    let vertices = [
        Vertex2D::new(0.0, -0.8, Color::RED),
        Vertex2D::new(-0.8, 0.8, Color::GREEN),
        Vertex2D::new(0.8, 0.8, Color::BLUE),
    ];

    let pixels = render_triangle(&device, 256, 256, Color::BLACK, vertices);
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/screenshots/rgb_triangle.png");
    save_png(&path, 256, 256, &pixels);
}

#[test]
fn generate_white_triangle() {
    let Some(device) = create_device() else {
        eprintln!("Skipping: no GPU available");
        return;
    };

    let vertices = [
        Vertex2D::new(0.0, -0.5, Color::WHITE),
        Vertex2D::new(-0.5, 0.5, Color::WHITE),
        Vertex2D::new(0.5, 0.5, Color::WHITE),
    ];

    let pixels = render_triangle(&device, 128, 128, Color::BLACK, vertices);
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/screenshots/white_triangle.png");
    save_png(&path, 128, 128, &pixels);
}

// ============================================================================
// Game of Life
// ============================================================================

const GOL_GRID_WIDTH: u32 = 128;
const GOL_GRID_HEIGHT: u32 = 128;
const GOL_CELL_COUNT: u32 = GOL_GRID_WIDTH * GOL_GRID_HEIGHT;

fn create_gol_initial_state() -> Vec<u32> {
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
            if (rng >> 32) % 4 == 0 {
                cells[(y * GOL_GRID_WIDTH + x) as usize] = 1;
            }
        }
    }

    cells
}

fn render_game_of_life(device: &Device, updates: u32) -> Vec<u8> {
    let render_width = 512u32;
    let render_height = 512u32;

    let compute_shader =
        ShaderModule::from_slang(device, include_str!("../shaders/game_of_life.slang"))
            .expect("Failed to load compute shader");

    let render_shader =
        ShaderModule::from_slang(device, include_str!("../shaders/game_of_life_render.slang"))
            .expect("Failed to load render shader");

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

    let mut use_buffer_a = true;
    let workgroups_x = (GOL_GRID_WIDTH + 7) / 8;
    let workgroups_y = (GOL_GRID_HEIGHT + 7) / 8;

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

#[test]
fn generate_game_of_life_50() {
    let Some(device) = create_device() else {
        eprintln!("Skipping: no GPU available");
        return;
    };

    let pixels = render_game_of_life(&device, 50);
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/screenshots/game_of_life_50.png");
    save_png(&path, 512, 512, &pixels);
}

#[test]
fn generate_game_of_life_100() {
    let Some(device) = create_device() else {
        eprintln!("Skipping: no GPU available");
        return;
    };

    let pixels = render_game_of_life(&device, 100);
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/screenshots/game_of_life_100.png");
    save_png(&path, 512, 512, &pixels);
}
