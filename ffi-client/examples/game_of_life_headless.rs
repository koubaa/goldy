//! Headless Game of Life — hybrid TaskGraph (compute + render) smoke test.
//!
//! Mirrors `goldy/ffi/tests/task_graph_game_of_life.rs` and `goldy/examples/game_of_life.rs`.
//!
//! Run from `goldy/ffi-client`: `cargo run --example game_of_life_headless`

use goldy_ffi_client::{
    Color, ComputePipeline, DeviceDescriptor, Instance, MosaicSlot, NodeAccess, RenderPipeline, RenderPipelineDesc,
    RenderTarget, RequestAdapterOptions, ResourceAccess, ResourceCategory, ResourceHandle, RetainedPool, ShaderModule,
    TaskGraph, TextureFormat,
};

const GRID_WIDTH: u32 = 128;
const GRID_HEIGHT: u32 = 128;
const CELL_COUNT: usize = (GRID_WIDTH * GRID_HEIGHT) as usize;

const SLOT_A: MosaicSlot = MosaicSlot(0);
const SLOT_B: MosaicSlot = MosaicSlot(1);

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
    println!("Goldy game_of_life_headless (ffi-client)\n");

    let instance = Instance::new()?;
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())?
        .request_device(&DeviceDescriptor::default())?;

    let initial = initial_cells();
    let mut retained_pool = RetainedPool::new(&device)?;
    let mut mosaic = retained_pool.mosaic()?;
    mosaic.emplace_pod::<u32>(&initial)?;
    mosaic.emplace_pod::<u32>(&initial)?;
    let cells = mosaic.build(&mut retained_pool)?;

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

    let target = RenderTarget::new(&device, GRID_WIDTH, GRID_HEIGHT, TextureFormat::Rgba8Unorm)?;

    let read_idx = cells.mosaic_view_resource_index(SLOT_A, ResourceAccess::ReadWrite)?;
    let write_idx = cells.mosaic_view_resource_index(SLOT_B, ResourceAccess::Write)?;

    let mut graph = TaskGraph::new();

    {
        let mut node = graph.compute_node("game_of_life", &compute_pipeline);
        node.bind_parcel_view(&cells, SLOT_A, NodeAccess::Read);
        node.bind_parcel_view(&cells, SLOT_B, NodeAccess::Write);
        node.bind_resources_raw(&[read_idx, write_idx]);
        node.dispatch(GRID_WIDTH.div_ceil(8), GRID_HEIGHT.div_ceil(8), 1);
    }

    let render_idx = cells.mosaic_view_resource_index(SLOT_B, ResourceAccess::ReadWrite)?;
    let mut pass = graph.render_pass("game_of_life_render", &target);
    pass.bind_parcel_view_mut(&cells, SLOT_B, NodeAccess::Read);
    pass.clear(Color::BLACK);
    pass.set_pipeline(&render_pipeline);
    pass.bind_resources_typed(&[ResourceHandle {
        category: ResourceCategory::Scattered,
        index: render_idx,
    }]);
    pass.draw_fullscreen();
    pass.finish_recorded();

    graph.dispatch(&device)?;

    let bytes = cells.mosaic_view_read_to_cpu(SLOT_B, &device)?;
    let cells_out: &[u32] = bytemuck::cast_slice(&bytes);
    assert_eq!(count_live(cells_out), 4, "still-life block should remain 4 live cells");

    let pixels = target.read_to_cpu()?;
    let cx = (GRID_WIDTH / 2) as usize;
    let cy = (GRID_HEIGHT / 2) as usize;
    let stride = (GRID_WIDTH * 4) as usize;
    let g = pixels[cy * stride + cx * 4 + 1];
    assert!(g > 100, "center pixel should show alive cells (g={g})");

    println!("Simulation + render OK ({GRID_WIDTH}x{GRID_HEIGHT}, {g} green at center)");
    Ok(())
}
