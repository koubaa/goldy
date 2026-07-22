//! Headless triangle example using goldy-ffi-client Scheme render pass + grant readback.
//!
//! Mirrors the dotnet/python headless triangle smoke tests.
//!
//! Run from `goldy/ffi-client`: `cargo run --example triangle_headless`

use goldy_ffi_client::{
    shader::builtins, BufferKind, Color, Context, DepthFormat, DeviceDescriptor, Instance, NodeAccess, RenderPipeline,
    RenderPipelineDesc, RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, TargetLoad, TextureFlags,
    TextureFormat, TextureKind, Vertex2D,
};

fn main() -> goldy_ffi_client::Result<()> {
    println!("Goldy triangle_headless (ffi-client / Scheme)\n");

    let instance = Instance::new()?;
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())?
        .request_device(&DeviceDescriptor::default())?;
    let ctx = Context::new(&device)?;

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
    let mut retained_pool = RetainedPool::new(&device)?;
    let vertex_buffer = retained_pool.acquire_buffer_with_data(&vertices, BufferKind::Scattered)?;

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;
    let readback = retained_pool.acquire_texture(
        WIDTH,
        HEIGHT,
        TextureFormat::Rgba8Unorm,
        TextureKind::Direct,
        TextureFlags::COPY_SRC.union(TextureFlags::COPY_DST),
        None,
    )?;

    let shader = ShaderModule::from_slang(&device, builtins::VERTEX_COLOR_2D)?;
    let pipeline = RenderPipeline::new(
        &device,
        &shader,
        &shader,
        &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        },
    )?;

    let mut scheme = Scheme::new(&ctx)?;
    let rt = scheme.lease_render_target(WIDTH, HEIGHT, TextureFormat::Rgba8Unorm, None::<DepthFormat>)?;
    {
        let mut pass = scheme.render_pass("triangle", &rt, TargetLoad::Clear(Color::BLACK));
        pass.with_buffer(&vertex_buffer, NodeAccess::Read);
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, &vertex_buffer);
        pass.draw(0..3, 0..1);
        pass.finish_recorded();
    }
    scheme.copy_to_texture(&rt, &readback)?;
    let grant = scheme.grant_read_texture(&readback)?;
    let submission = scheme.submit()?;
    let pixels = grant.consume(&submission)?;
    assert!(pixels.iter().any(|&b| b > 0), "readback should contain lit pixels");

    println!("Triangle rendered and read back successfully ({} bytes).", pixels.len());
    Ok(())
}
