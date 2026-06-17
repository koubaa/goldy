//! Headless integration test: hybrid compute + raster via TaskGraph FFI.
//!
//! Mirrors `goldy/examples/game_of_life.rs` (one simulation step + render readback).

mod common;

use common::{last_ffi_message, open_device};
use goldy_ffi::{
    goldy_compute_pipeline_create, goldy_compute_pipeline_destroy, goldy_device_destroy, goldy_instance_destroy,
    goldy_mosaic_builder_build, goldy_mosaic_builder_create, goldy_mosaic_builder_emplace, goldy_parcel_destroy,
    goldy_parcel_mosaic_view_read_to_cpu, goldy_parcel_mosaic_view_resource_index, goldy_parcel_mosaic_view_size,
    goldy_render_pipeline_create, goldy_render_pipeline_destroy, goldy_render_target_buffer_size,
    goldy_render_target_create, goldy_render_target_destroy, goldy_render_target_read_to_buffer,
    goldy_retained_pool_create, goldy_retained_pool_destroy, goldy_shader_create, goldy_shader_destroy,
    goldy_task_graph_compute_node_begin, goldy_task_graph_compute_node_with_parcel_view,
    goldy_task_graph_compute_node_with_resource_slots, goldy_task_graph_compute_node_dispatch, goldy_task_graph_create,
    goldy_task_graph_destroy, goldy_task_graph_dispatch, goldy_task_graph_render_pass_begin,
    goldy_task_graph_render_pass_with_parcel_view, goldy_task_graph_render_pass_with_views,
    goldy_task_graph_render_pass_clear, goldy_task_graph_render_pass_draw_fullscreen,
    goldy_task_graph_render_pass_finish, goldy_task_graph_render_pass_set_pipeline, GoldyColor, GoldyNodeAccess,
    GoldyRenderPipelineDesc, GoldyResourceAccess, GoldyResult, GoldyTextureFormat,
};
use std::ffi::CString;

const GRID_WIDTH: u32 = 128;
const GRID_HEIGHT: u32 = 128;
const CELL_COUNT: usize = (GRID_WIDTH * GRID_HEIGHT) as usize;
const SLOT_A: u32 = 0;
const SLOT_B: u32 = 1;

const COMPUTE_SHADER: &str = include_str!("../../shaders/game_of_life.slang");
const RENDER_SHADER: &str = include_str!("../../shaders/game_of_life_render.slang");

/// Stable 2×2 block at the grid center (still life).
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

#[test]
fn task_graph_game_of_life_hybrid_simulate_and_render() {
    unsafe {
        let (instance, device) = open_device();

        let initial = initial_cells();
        let cell_bytes = std::mem::size_of::<u32>();

        let pool = goldy_retained_pool_create(device);
        assert!(!pool.is_null(), "{}", last_ffi_message());

        let mosaic = goldy_mosaic_builder_create();
        assert!(!mosaic.is_null(), "{}", last_ffi_message());

        assert_eq!(
            goldy_mosaic_builder_emplace(
                mosaic,
                initial.as_ptr() as *const u8,
                initial.len() * cell_bytes,
                CELL_COUNT as u64,
                cell_bytes as u32,
            ),
            SLOT_A,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_mosaic_builder_emplace(
                mosaic,
                initial.as_ptr() as *const u8,
                initial.len() * cell_bytes,
                CELL_COUNT as u64,
                cell_bytes as u32,
            ),
            SLOT_B,
            "{}",
            last_ffi_message()
        );

        let cells = goldy_mosaic_builder_build(mosaic, pool);
        assert!(!cells.is_null(), "{}", last_ffi_message());

        let compute_src = CString::new(COMPUTE_SHADER).unwrap();
        let compute_shader = goldy_shader_create(device, compute_src.as_ptr());
        assert!(!compute_shader.is_null(), "{}", last_ffi_message());

        let render_src = CString::new(RENDER_SHADER).unwrap();
        let render_shader = goldy_shader_create(device, render_src.as_ptr());
        assert!(!render_shader.is_null(), "{}", last_ffi_message());

        let compute_pipeline = goldy_compute_pipeline_create(device, compute_shader);
        assert!(!compute_pipeline.is_null(), "{}", last_ffi_message());

        let render_desc = GoldyRenderPipelineDesc {
            vertex_stride: 0,
            target_format: GoldyTextureFormat::Rgba8Unorm,
            ..Default::default()
        };
        let render_pipeline = goldy_render_pipeline_create(device, render_shader, render_shader, &render_desc);
        assert!(!render_pipeline.is_null(), "{}", last_ffi_message());

        let target = goldy_render_target_create(device, GRID_WIDTH, GRID_HEIGHT, GoldyTextureFormat::Rgba8Unorm);
        assert!(!target.is_null(), "{}", last_ffi_message());

        let read_idx = goldy_parcel_mosaic_view_resource_index(cells, SLOT_A, GoldyResourceAccess::ReadWrite);
        let write_idx = goldy_parcel_mosaic_view_resource_index(cells, SLOT_B, GoldyResourceAccess::Write);
        assert_ne!(read_idx, u32::MAX);
        assert_ne!(write_idx, u32::MAX);

        let graph = goldy_task_graph_create();
        assert!(!graph.is_null());

        let compute_label = CString::new("game_of_life").unwrap();
        assert_eq!(
            goldy_task_graph_compute_node_begin(graph, compute_label.as_ptr(), compute_pipeline),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_compute_node_with_parcel_view(graph, cells, SLOT_A, GoldyNodeAccess::Read),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_compute_node_with_parcel_view(graph, cells, SLOT_B, GoldyNodeAccess::Write),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        let slots = [read_idx, write_idx];
        assert_eq!(
            goldy_task_graph_compute_node_with_resource_slots(graph, slots.as_ptr(), 2),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_compute_node_dispatch(graph, GRID_WIDTH.div_ceil(8), GRID_HEIGHT.div_ceil(8), 1),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let render_idx = goldy_parcel_mosaic_view_resource_index(cells, SLOT_B, GoldyResourceAccess::ReadWrite);
        assert_ne!(render_idx, u32::MAX);

        let render_label = CString::new("game_of_life_render").unwrap();
        assert_eq!(
            goldy_task_graph_render_pass_begin(graph, render_label.as_ptr(), target),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_render_pass_with_parcel_view(graph, cells, SLOT_B, GoldyNodeAccess::Read),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_render_pass_clear(
                graph,
                GoldyColor {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                }
            ),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_render_pass_set_pipeline(graph, render_pipeline),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        let typed = [0u32, render_idx];
        assert_eq!(
            goldy_task_graph_render_pass_with_views(graph, typed.as_ptr(), 1),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_render_pass_draw_fullscreen(graph),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_render_pass_finish(graph),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_task_graph_dispatch(graph, device),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let view_size = goldy_parcel_mosaic_view_size(cells, SLOT_B) as usize;
        let mut readback = vec![0u8; view_size];
        assert_eq!(
            goldy_parcel_mosaic_view_read_to_cpu(cells, SLOT_B, device, readback.as_mut_ptr(), readback.len()),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        let cells_out: &[u32] =
            std::slice::from_raw_parts(readback.as_ptr() as *const u32, readback.len() / cell_bytes);
        assert_eq!(
            count_live(cells_out),
            4,
            "still-life block should remain 4 live cells after one step"
        );

        let px_size = goldy_render_target_buffer_size(target);
        let mut pixels = vec![0u8; px_size];
        assert_eq!(
            goldy_render_target_read_to_buffer(target, pixels.as_mut_ptr(), pixels.len()),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let cx = (GRID_WIDTH / 2) as usize;
        let cy = (GRID_HEIGHT / 2) as usize;
        let stride = (GRID_WIDTH * 4) as usize;
        let i = cy * stride + cx * 4;
        let g = pixels[i + 1];
        assert!(
            g > 100,
            "center pixel should show alive green channel, got g={g} at ({cx},{cy})"
        );

        goldy_task_graph_destroy(graph);
        goldy_render_pipeline_destroy(render_pipeline);
        goldy_compute_pipeline_destroy(compute_pipeline);
        goldy_shader_destroy(render_shader);
        goldy_shader_destroy(compute_shader);
        goldy_render_target_destroy(target);
        goldy_parcel_destroy(cells);
        goldy_retained_pool_destroy(pool);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}
