//! Integration tests for RenderTarget with real GPU.
//!
//! These tests require a GPU and are skipped in CI if no GPU is available.
#![cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]

use goldy::{
    Buffer, Color, CommandEncoder, CompareFunction, DataAccess, DepthFormat, DepthStencilState,
    DeviceDescriptor, IndexFormat, Instance, PrimitiveTopology, RenderPipeline, RenderPipelineDesc,
    RenderTarget, RequestAdapterOptions, ShaderModule, TextureFormat, Vertex2D, VertexAttribute,
    VertexBufferLayout, VertexFormat,
};

fn create_device() -> Option<goldy::Device> {
    let instance = Instance::new().ok()?;
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .ok()?
        .request_device(&DeviceDescriptor::default())
        .ok()
}

#[test]
fn test_vulkan_render_target_creation() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let target = RenderTarget::new(&device, 800, 600, TextureFormat::Rgba8Unorm)
        .expect("Failed to create render target");

    assert_eq!(target.width(), 800);
    assert_eq!(target.height(), 600);
    assert_eq!(target.format(), TextureFormat::Rgba8Unorm);
    assert_eq!(target.buffer_size(), 800 * 600 * 4);
}

#[test]
fn test_vulkan_render_and_readback() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let target = RenderTarget::new(&device, 100, 100, TextureFormat::Rgba8Unorm)
        .expect("Failed to create render target");

    // Create a simple shader
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

    let shader = ShaderModule::from_slang(&device, shader_source).expect("Failed to create shader");

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
    .expect("Failed to create pipeline");

    // Create a simple triangle
    let vertices = [
        Vertex2D::new(0.0, -0.5, Color::RED),
        Vertex2D::new(-0.5, 0.5, Color::GREEN),
        Vertex2D::new(0.5, 0.5, Color::BLUE),
    ];
    let vertex_buffer = Buffer::with_data(&device, &vertices, DataAccess::Scattered)
        .expect("Failed to create vertex buffer");

    // Render
    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(Color::BLACK);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertex_buffer);
        pass.draw(0..3, 0..1);
    }

    target.render(encoder).expect("Failed to render");

    // Read back
    let pixels = target.read_to_cpu().expect("Failed to read pixels");

    assert_eq!(pixels.len(), 100 * 100 * 4);

    // The center of the image should have some non-black pixels (the triangle)
    // Check that not all pixels are black
    let has_non_black = pixels.chunks(4).any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0);
    assert!(
        has_non_black,
        "Expected rendered triangle to have non-black pixels"
    );
}

#[test]
fn test_render_target_clear_only() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let target = RenderTarget::new(&device, 4, 4, TextureFormat::Rgba8Unorm)
        .expect("Failed to create render target");

    // Just clear to a solid color
    let clear_color = Color::from_rgb(128, 64, 32);
    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(clear_color);
    }

    target.render(encoder).expect("Failed to render");

    let pixels = target.read_to_cpu().expect("Failed to read pixels");

    // All pixels should be the clear color
    for chunk in pixels.chunks(4) {
        assert_eq!(chunk[0], 128, "Red channel mismatch");
        assert_eq!(chunk[1], 64, "Green channel mismatch");
        assert_eq!(chunk[2], 32, "Blue channel mismatch");
        assert_eq!(chunk[3], 255, "Alpha channel mismatch");
    }
}

#[test]
fn test_multiple_render_targets() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    // Create multiple render targets
    let target1 = RenderTarget::new(&device, 10, 10, TextureFormat::Rgba8Unorm)
        .expect("Failed to create target 1");
    let target2 = RenderTarget::new(&device, 20, 20, TextureFormat::Rgba8Unorm)
        .expect("Failed to create target 2");

    // Render to both with different colors
    let mut encoder1 = CommandEncoder::new();
    {
        let mut pass = encoder1.begin_render_pass();
        pass.clear(Color::RED);
    }
    target1
        .render(encoder1)
        .expect("Failed to render to target 1");

    let mut encoder2 = CommandEncoder::new();
    {
        let mut pass = encoder2.begin_render_pass();
        pass.clear(Color::BLUE);
    }
    target2
        .render(encoder2)
        .expect("Failed to render to target 2");

    // Read back and verify
    let pixels1 = target1.read_to_cpu().expect("Failed to read target 1");
    let pixels2 = target2.read_to_cpu().expect("Failed to read target 2");

    // Target 1 should be red
    assert_eq!(pixels1[0], 255); // R
    assert_eq!(pixels1[1], 0); // G
    assert_eq!(pixels1[2], 0); // B

    // Target 2 should be blue
    assert_eq!(pixels2[0], 0); // R
    assert_eq!(pixels2[1], 0); // G
    assert_eq!(pixels2[2], 255); // B
}

#[test]
fn test_indexed_drawing() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let target = RenderTarget::new(&device, 100, 100, TextureFormat::Rgba8Unorm)
        .expect("Failed to create render target");

    // Create a simple shader
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

    let shader = ShaderModule::from_slang(&device, shader_source).expect("Failed to create shader");

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
    .expect("Failed to create pipeline");

    // Create a quad using 4 vertices and 6 indices (2 triangles)
    let vertices = [
        Vertex2D::new(-0.5, -0.5, Color::RED),  // 0: bottom-left
        Vertex2D::new(0.5, -0.5, Color::GREEN), // 1: bottom-right
        Vertex2D::new(0.5, 0.5, Color::BLUE),   // 2: top-right
        Vertex2D::new(-0.5, 0.5, Color::WHITE), // 3: top-left
    ];
    let vertex_buffer = Buffer::with_data(&device, &vertices, DataAccess::Scattered)
        .expect("Failed to create vertex buffer");

    // Indices for two triangles forming a quad
    let indices: [u16; 6] = [
        0, 1, 2, // First triangle
        0, 2, 3, // Second triangle
    ];
    let index_buffer = Buffer::with_data(&device, &indices, DataAccess::Scattered)
        .expect("Failed to create index buffer");

    // Render using indexed drawing
    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(Color::BLACK);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertex_buffer);
        pass.set_index_buffer(&index_buffer, IndexFormat::Uint16);
        pass.draw_indexed(0..6, 0, 0..1);
    }

    target.render(encoder).expect("Failed to render");

    // Read back and verify we got something rendered
    let pixels = target.read_to_cpu().expect("Failed to read pixels");

    assert_eq!(pixels.len(), 100 * 100 * 4);

    // The quad should cover a significant portion of the image
    // Check that we have non-black pixels (the quad was drawn)
    let non_black_count = pixels
        .chunks(4)
        .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
        .count();
    assert!(
        non_black_count > 1000,
        "Expected quad to cover at least 1000 pixels, got {}",
        non_black_count
    );
}

#[test]
fn test_indexed_drawing_uint32() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let target = RenderTarget::new(&device, 50, 50, TextureFormat::Rgba8Unorm)
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

    let shader = ShaderModule::from_slang(&device, shader_source).expect("Failed to create shader");

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
    .expect("Failed to create pipeline");

    // Simple triangle with u32 indices
    let vertices = [
        Vertex2D::new(0.0, -0.8, Color::RED),
        Vertex2D::new(-0.8, 0.8, Color::RED),
        Vertex2D::new(0.8, 0.8, Color::RED),
    ];
    let vertex_buffer = Buffer::with_data(&device, &vertices, DataAccess::Scattered)
        .expect("Failed to create vertex buffer");

    // Use u32 indices
    let indices: [u32; 3] = [0, 1, 2];
    let index_buffer = Buffer::with_data(&device, &indices, DataAccess::Scattered)
        .expect("Failed to create index buffer");

    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(Color::BLACK);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertex_buffer);
        pass.set_index_buffer(&index_buffer, IndexFormat::Uint32);
        pass.draw_indexed(0..3, 0, 0..1);
    }

    target.render(encoder).expect("Failed to render");

    let pixels = target.read_to_cpu().expect("Failed to read pixels");

    // Should have red pixels from the triangle
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

// ============================================================================
// Depth occlusion tests
//
// These tests assert known pixel colors directly rather than comparing against
// a self-generated reference image.  A self-generated reference can mask bugs:
// if depth is silently disabled, both generation and comparison produce the
// same wrong color, FLIP = 0, and the test "passes" despite being incorrect.
// Hard-coded color assertions catch that class of bug immediately.
// ============================================================================

/// Vertex type carrying (x, y, z) position and RGBA color.
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

/// Depth occlusion: near geometry (red, z=0.2) must block far geometry (green,
/// z=0.6) even when the green quad is submitted to the GPU second.
///
/// Expected output: every pixel is red.
/// If depth testing is disabled the green quad overwrites the red one and
/// every pixel is green — this test catches that regression.
#[test]
fn test_depth_occlusion_red_beats_green() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let target = RenderTarget::new_with_depth(
        &device,
        64,
        64,
        TextureFormat::Rgba8Unorm,
        Some(DepthFormat::Depth32Float),
    )
    .expect("Failed to create render target with depth");

    let shader_source = include_str!("../shaders/depth_test.slang");
    let shader =
        ShaderModule::from_slang(&device, shader_source).expect("Failed to compile shader");

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
    .expect("Failed to create depth pipeline");

    // Large-triangle trick: a single triangle that covers all of NDC [-1,1]².
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

    // Red (z=0.2, near) drawn first; green (z=0.6, far) drawn second.
    // Without depth testing green would overwrite red — this is the regression
    // we are guarding against.
    let red_verts = make_tri(0.2, [1.0, 0.0, 0.0, 1.0]);
    let green_verts = make_tri(0.6, [0.0, 1.0, 0.0, 1.0]);

    let red_vb = Buffer::with_data(&device, &red_verts, DataAccess::Scattered)
        .expect("Failed to create red VB");
    let green_vb = Buffer::with_data(&device, &green_verts, DataAccess::Scattered)
        .expect("Failed to create green VB");

    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(Color::BLACK);
        pass.clear_depth(1.0);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &red_vb);
        pass.draw(0..3, 0..1);
        pass.set_vertex_buffer(0, &green_vb);
        pass.draw(0..3, 0..1);
    }

    target.render(encoder).expect("Failed to render");
    let pixels = target.read_to_cpu().expect("Failed to read pixels");

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
        "Depth occlusion failed: expected all {} pixels red, got {} red / {} green.\n\
         This means depth testing is not working — green (drawn second, z=0.6) \
         overwrote red (z=0.2) instead of being occluded.",
        total,
        red_count,
        green_count
    );
}

/// Depth occlusion (reversed): far geometry (red, z=0.8) must lose to near
/// geometry (green, z=0.2) even when the red quad is submitted first.
///
/// Expected output: every pixel is green.
/// This is the complement of the test above and verifies that a late-drawn
/// near fragment correctly overwrites an early-drawn far fragment.
#[test]
fn test_depth_occlusion_green_beats_red() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let target = RenderTarget::new_with_depth(
        &device,
        64,
        64,
        TextureFormat::Rgba8Unorm,
        Some(DepthFormat::Depth32Float),
    )
    .expect("Failed to create render target with depth");

    let shader_source = include_str!("../shaders/depth_test.slang");
    let shader =
        ShaderModule::from_slang(&device, shader_source).expect("Failed to compile shader");

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

    // Red (z=0.8, far) drawn first; green (z=0.2, near) drawn second.
    // Green must win because it is closer, not because it is drawn last.
    let red_verts = make_tri(0.8, [1.0, 0.0, 0.0, 1.0]);
    let green_verts = make_tri(0.2, [0.0, 1.0, 0.0, 1.0]);

    let red_vb = Buffer::with_data(&device, &red_verts, DataAccess::Scattered)
        .expect("Failed to create red VB");
    let green_vb = Buffer::with_data(&device, &green_verts, DataAccess::Scattered)
        .expect("Failed to create green VB");

    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(Color::BLACK);
        pass.clear_depth(1.0);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &red_vb);
        pass.draw(0..3, 0..1);
        pass.set_vertex_buffer(0, &green_vb);
        pass.draw(0..3, 0..1);
    }

    target.render(encoder).expect("Failed to render");
    let pixels = target.read_to_cpu().expect("Failed to read pixels");

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
        "Depth occlusion failed: expected all {} pixels green, got {} green / {} red.\n\
         This means depth testing is not working — the near green fragment (z=0.2) \
         did not overwrite the far red fragment (z=0.8).",
        total,
        green_count,
        red_count
    );
}

/// Render a fullscreen triangle whose fragment shader reads a value from a bindless buffer
/// via resource bindings. Verifies the global argument buffer is correctly bound to offscreen
/// render targets (a bug that produces a completely blank output if missing).
#[test]
fn test_render_target_bindless_buffer_read() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let data = vec![1u32; 4];
    let buffer = Buffer::with_data(&device, &data, DataAccess::Scattered).expect("create buffer");

    // Fragment shader reads buffer[0] via bindless resource binding.
    // Outputs bright green when value == 1 (alive), dark gray otherwise.
    // Both branches are visually distinct from the black clear color, so we can
    // tell whether the shader ran vs. the draw call being skipped entirely.
    // If the argument buffer is not bound the GPU reads 0 → dark gray pixels.
    // Mirrors the GOL render shader structure to avoid platform-specific
    // Slang codegen differences (UV varying, local variable assignment).
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

    let shader = ShaderModule::from_slang(&device, shader_source).expect("compile bindless shader");

    let target =
        RenderTarget::new(&device, 4, 4, TextureFormat::Rgba8Unorm).expect("create target");

    let pipeline = RenderPipeline::new(
        &device,
        &shader,
        &shader,
        &RenderPipelineDesc {
            // Empty vertex layout: this shader uses SV_VertexID with no vertex
            // attributes, so no vertex descriptor should be set on the pipeline.
            vertex_layout: VertexBufferLayout {
                attributes: vec![],
                stride: 0,
            },
            topology: PrimitiveTopology::TriangleList,
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        },
    )
    .expect("create pipeline");

    let mut encoder = CommandEncoder::new();
    {
        let mut pass = encoder.begin_render_pass();
        pass.clear(Color::BLACK);
        pass.set_pipeline(&pipeline);
        pass.bind_resources(&[&buffer]);
        pass.draw(0..3, 0..1);
    }
    target.render(encoder).expect("render");

    let pixels = target.read_to_cpu().expect("readback");
    assert_eq!(pixels.len(), 4 * 4 * 4);

    // Every pixel should be the "alive" green color (val==1 branch).
    // Clear color:  black  → [  0,   0,   0, 255]  (draw call never ran)
    // val==0 branch: dark  → [ 12,  20,  25, 255]  (argument buffer not bound)
    // val==1 branch: green → [ 51, 229,  76, 255]  (correct)
    // Checking G > 100 distinguishes the green branch from both failure modes.
    let all_green = pixels.chunks(4).all(|p| p[1] > 100);
    assert!(
        all_green,
        "Expected all pixels green (alive) from bindless buffer read.\n\
         First pixel: {:?}\n\
         black=[0,0,0,255] means the draw call never executed;\n\
         dark=[12,20,25,255] means the argument buffer was not bound to the render encoder.",
        &pixels[..4]
    );
}
