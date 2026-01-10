//! Integration tests for RenderTarget with real GPU.
//!
//! These tests require a GPU and are skipped in CI if no GPU is available.

use rag::{
    Buffer, BufferUsage, Color, CommandEncoder, DeviceType, Instance, RenderPipeline,
    RenderPipelineDesc, RenderTarget, ShaderModule, TextureFormat, Vertex2D,
};

fn create_device() -> Option<rag::Device> {
    let instance = Instance::new().ok()?;
    instance.create_device(DeviceType::DiscreteGpu).ok()
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

    let shader = ShaderModule::from_slang(&device, shader_source)
        .expect("Failed to create shader");

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
    let vertex_buffer = Buffer::with_data(&device, &vertices, BufferUsage::VERTEX)
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
    assert!(has_non_black, "Expected rendered triangle to have non-black pixels");
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
    target1.render(encoder1).expect("Failed to render to target 1");

    let mut encoder2 = CommandEncoder::new();
    {
        let mut pass = encoder2.begin_render_pass();
        pass.clear(Color::BLUE);
    }
    target2.render(encoder2).expect("Failed to render to target 2");

    // Read back and verify
    let pixels1 = target1.read_to_cpu().expect("Failed to read target 1");
    let pixels2 = target2.read_to_cpu().expect("Failed to read target 2");

    // Target 1 should be red
    assert_eq!(pixels1[0], 255); // R
    assert_eq!(pixels1[1], 0);   // G
    assert_eq!(pixels1[2], 0);   // B

    // Target 2 should be blue
    assert_eq!(pixels2[0], 0);   // R
    assert_eq!(pixels2[1], 0);   // G
    assert_eq!(pixels2[2], 255); // B
}


