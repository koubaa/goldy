//! Headless Game of Life — hybrid Scheme (compute + render) smoke test.
//!
//! Mirrors native scheme patterns: one retained record buffer (`"a"` / `"b"`), field
//! parcels bound via `with_parcel`.
//!
//! Run from `goldy/ffi-client`: `cargo run --example game_of_life_headless`

use goldy_ffi_client::{
    Color, ComputePipeline, Context, DepthFormat, DeviceDescriptor, Instance, NodeAccess, RenderPipeline,
    RenderPipelineDesc, RequestAdapterOptions, ResourceAccess, RetainedPool, Scheme, ShaderModule, TextureFlags,
    TextureFormat, TextureKind,
};

const GRID_WIDTH: u32 = 128;
const GRID_HEIGHT: u32 = 128;
const CELL_COUNT: usize = (GRID_WIDTH * GRID_HEIGHT) as usize;

const COMPUTE_SHADER: &str = include_str!("../../shaders/game_of_life.slang");
const RENDER_SHADER: &str = include_str!("../../shaders/game_of_life_render.slang");

fn initial_cells() -> Vec<u32> {
    let mut cells = vec![0u32; CELL_COUNT];
    for y in 63..=64 {
        for x in 63..=64 {
            cells[(y * GRID_WIDTH + x) as usize] = 1;
        }
    }
    cells
}

fn count_live(cells: &[u32]) -> u32 {
    cells.iter().filter(|&&c| c == 1).count() as u32
}

fn main() -> goldy_ffi_client::Result<()> {
    println!("Goldy game_of_life_headless (ffi-client / Scheme)\n");

    let instance = Instance::new()?;
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())?
        .request_device(&DeviceDescriptor::default())?;
    let ctx = Context::new(&device)?;

    let initial = initial_cells();
    let mut retained_pool = RetainedPool::new(&device)?;
    let cells = retained_pool.acquire_record_pod(&[("a", &initial), ("b", &initial)])?;

    let compute_shader = ShaderModule::from_slang(&device, COMPUTE_SHADER)?;
    let render_shader = ShaderModule::from_slang(&device, RENDER_SHADER)?;
    let compute_pipeline = ComputePipeline::new(&device, &compute_shader)?;
    let render_pipeline = RenderPipeline::new(
        &device,
        &render_shader,
        &render_shader,
        &RenderPipelineDesc {
            target_format: TextureFormat::Rgba8Unorm,
            ..Default::default()
        },
    )?;

    let readback = retained_pool.acquire_texture(
        GRID_WIDTH,
        GRID_HEIGHT,
        TextureFormat::Rgba8Unorm,
        TextureKind::Direct,
        TextureFlags::COPY_SRC.union(TextureFlags::COPY_DST),
        None,
    )?;

    let read = cells.field(0)?;
    let write = cells.field(1)?;
    let mut scheme = Scheme::new(&ctx)?;

    {
        let mut node = scheme.compute_node("game_of_life", &compute_pipeline);
        node.with_parcel(&read, NodeAccess::Read, ResourceAccess::ReadWrite);
        node.with_parcel(&write, NodeAccess::Write, ResourceAccess::Write);
        node.dispatch(GRID_WIDTH.div_ceil(8), GRID_HEIGHT.div_ceil(8), 1);
    }

    let rt = scheme.lease_render_target(GRID_WIDTH, GRID_HEIGHT, TextureFormat::Rgba8Unorm, None::<DepthFormat>)?;
    {
        let current = cells.field(1)?;
        let mut pass = scheme.render_pass("game_of_life_render", &rt);
        pass.with_parcel(&current, NodeAccess::Read);
        pass.clear(Color::BLACK);
        pass.set_pipeline(&render_pipeline);
        pass.draw_fullscreen();
        pass.finish_recorded();
    }

    scheme.copy_to_texture(&rt, &readback)?;
    let grant = scheme.grant_read_texture(&readback)?;
    let submission = scheme.submit()?;
    let pixels = grant.consume(&submission)?;

    let bytes = cells.unit_read_to_cpu(1, &device)?;
    let cells_out: &[u32] = bytemuck::cast_slice(&bytes);
    assert_eq!(count_live(cells_out), 4, "still-life block should remain 4 live cells");

    let cx = (GRID_WIDTH / 2) as usize;
    let cy = (GRID_HEIGHT / 2) as usize;
    let stride = (GRID_WIDTH * 4) as usize;
    let g = pixels[cy * stride + cx * 4 + 1];
    assert!(g > 100, "center pixel should show alive cells (g={g})");

    println!("Simulation + render OK ({GRID_WIDTH}x{GRID_HEIGHT}, {g} green at center)");
    Ok(())
}
