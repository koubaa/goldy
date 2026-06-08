//! Headless triangle example using goldy-ffi-client TaskGraph.
//!
//! Mirrors the dotnet/python headless triangle smoke tests.
//!
//! Run from `goldy/ffi-client`: `cargo run --example triangle_headless`

use goldy_ffi_client::{
    shader::builtins, BufferKind, Color, DeviceDescriptor, Instance, NodeAccess, RenderPipeline,
    RenderPipelineDesc, RenderTarget, RequestAdapterOptions, ShaderModule, TaskGraph, Vertex2D,
};

fn main() -> goldy_ffi_client::Result<()> {
    println!("Goldy triangle_headless (ffi-client)\n");

    let instance = Instance::new()?;
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())?
        .request_device(&DeviceDescriptor::default())?;

    let vertices = [
        Vertex2D {
            position: [0.0, -0.5],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex2D {
            position: [-0.5, 0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex2D {
            position: [0.5, 0.5],
            color: [0.0, 0.0, 1.0, 1.0],
        },
    ];
    let vertex_buffer = device.alloc_buffer_with_data(&vertices, BufferKind::Scattered)?;

    let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
    let pipeline = RenderPipeline::new(
        &device,
        &shader,
        &shader,
        &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: goldy_ffi_client::TextureFormat::Rgba8Unorm,
            ..Default::default()
        },
    )?;

    let target = RenderTarget::new(&device, 64, 64, goldy_ffi_client::TextureFormat::Rgba8Unorm)?;

    let mut graph = TaskGraph::new();
    let mut pass = graph.render_pass("triangle", &target);
    pass.bind_buffer_mut(&vertex_buffer, NodeAccess::Read);
    pass.clear(Color::BLACK);
    pass.set_pipeline(&pipeline);
    pass.set_vertex_buffer(0, &vertex_buffer);
    pass.draw(0..3, 0..1);
    pass.finish_recorded();
    graph.dispatch(&device)?;

    let pixels = target.read_to_cpu()?;
    assert!(pixels.iter().any(|&b| b > 0), "readback should contain lit pixels");

    println!("Triangle rendered and read back successfully ({} bytes).", pixels.len());
    Ok(())
}
