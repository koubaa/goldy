//! Integration tests for the windowed surface graph path (render pass + swapchain copy).
//!
//! Legacy TaskGraph path. Scheme coverage: `scheme_render_integration.rs`.
//! Delete this file when ekrano migrates (Phase 2).
//!
//! Full `Surface::submit_graph_to_frame` requires a live WSI window (see `examples/triangle.rs`).
//! These tests exercise the same render-pass graph submission against an offscreen
//! `RenderTarget` and verify pixels via CPU readback.
#![cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]

use goldy::{
    shader::builtins, BufferKind, Color, DeviceDescriptor, Instance, NodeAccess, RenderPipeline, RenderPipelineDesc,
    RenderTarget, RequestAdapterOptions, ShaderModule, TaskGraph, TextureFormat, Vertex2D,
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
}

fn make_device() -> Option<goldy::Device> {
    let instance = Instance::new().ok()?;
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .ok()?
        .request_device(&DeviceDescriptor::default())
        .ok()
}

/// Same graph shape as `examples/triangle.rs` (render_pass → submit), without swapchain acquire.
// Legacy TaskGraph — migrated: `scheme_render_pass_triangle_readback`
#[test]
fn render_pass_task_graph_triangle_readback() {
    let Some(device) = make_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };
    let ctx = device.create_context().expect("context");

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

    let target = RenderTarget::new(&device, W, H, TextureFormat::Rgba8Unorm).expect("render target");
    let vertex_buffer = test_alloc_buffer_with_data(&device, &vertices, BufferKind::Scattered);
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

    let mut graph = TaskGraph::new();
    let mut pass = graph.render_pass("triangle", &target);
    pass.with_buffer(&vertex_buffer, NodeAccess::Read);
    pass.clear(clear);
    pass.set_pipeline(&pipeline);
    pass.set_vertex_buffer(0, &vertex_buffer);
    pass.draw(0..3, 0..1);
    pass.finish_recorded();
    graph.submit(&ctx).expect("graph submit");

    let pixels = target.read_to_cpu().expect("readback");
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
