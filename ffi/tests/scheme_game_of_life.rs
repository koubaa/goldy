//! Headless integration test: hybrid compute + raster via Scheme FFI.

mod common;

use common::{last_ffi_message, open_device};
use goldy_ffi::{
    goldy_buffer_destroy, goldy_buffer_unit_byte_size, goldy_buffer_unit_read_to_cpu, goldy_compute_pipeline_create,
    goldy_compute_pipeline_destroy, goldy_context_create, goldy_context_destroy, goldy_device_destroy,
    goldy_instance_destroy, goldy_read_grant_consume, goldy_read_grant_destroy, goldy_record_builder_build,
    goldy_record_builder_create, goldy_record_builder_emplace, goldy_render_pipeline_create,
    goldy_render_pipeline_destroy, goldy_retained_pool_acquire_texture, goldy_retained_pool_create,
    goldy_retained_pool_destroy, goldy_scheme_compute_node_begin, goldy_scheme_compute_node_dispatch,
    goldy_scheme_compute_node_with_field, goldy_scheme_copy_to_texture, goldy_scheme_create, goldy_scheme_destroy,
    goldy_scheme_grant_read_texture, goldy_scheme_lease_render_target, goldy_scheme_render_pass_begin,
    goldy_scheme_render_pass_draw_fullscreen, goldy_scheme_render_pass_finish, goldy_scheme_render_pass_set_pipeline,
    goldy_scheme_render_pass_with_field, goldy_scheme_render_target_lease_destroy, goldy_scheme_submission_destroy,
    goldy_scheme_submit, goldy_shader_create, goldy_shader_destroy, goldy_texture_destroy, GoldyColor,
    GoldyDepthFormat, GoldyNodeAccess, GoldyRenderPipelineDesc, GoldyResult, GoldyTargetLoad, GoldyTextureFlags,
    GoldyTextureFormat, GoldyTextureKind,
};
use std::ffi::CString;

const GRID_WIDTH: u32 = 128;
const GRID_HEIGHT: u32 = 128;
const CELL_COUNT: usize = (GRID_WIDTH * GRID_HEIGHT) as usize;
const SLOT_A: u32 = 0;
const SLOT_B: u32 = 1;

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

#[test]
fn scheme_game_of_life_hybrid_simulate_and_render() {
    unsafe {
        let (instance, device) = open_device();

        let ctx = goldy_context_create(device);
        assert!(!ctx.is_null(), "{}", last_ffi_message());

        let initial = initial_cells();
        let cell_bytes = std::mem::size_of::<u32>();

        let pool = goldy_retained_pool_create(device);
        assert!(!pool.is_null(), "{}", last_ffi_message());

        let builder = goldy_record_builder_create();
        assert!(!builder.is_null(), "{}", last_ffi_message());

        let name_a = CString::new("a").unwrap();
        let name_b = CString::new("b").unwrap();
        assert_eq!(
            goldy_record_builder_emplace(
                builder,
                name_a.as_ptr(),
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
            goldy_record_builder_emplace(
                builder,
                name_b.as_ptr(),
                initial.as_ptr() as *const u8,
                initial.len() * cell_bytes,
                CELL_COUNT as u64,
                cell_bytes as u32,
            ),
            SLOT_B,
            "{}",
            last_ffi_message()
        );

        let cells = goldy_record_builder_build(builder, pool);
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

        let readback = goldy_retained_pool_acquire_texture(
            pool,
            GRID_WIDTH,
            GRID_HEIGHT,
            GoldyTextureFormat::Rgba8Unorm,
            GoldyTextureKind::Direct,
            GoldyTextureFlags(GoldyTextureFlags::COPY_SRC.0 | GoldyTextureFlags::COPY_DST.0),
            std::ptr::null(),
            0,
        );
        assert!(!readback.is_null(), "{}", last_ffi_message());

        let scheme = goldy_scheme_create(ctx);
        assert!(!scheme.is_null());

        let compute_label = CString::new("game_of_life").unwrap();
        assert_eq!(
            goldy_scheme_compute_node_begin(scheme, compute_label.as_ptr(), compute_pipeline),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_compute_node_with_field(scheme, cells, SLOT_A, GoldyNodeAccess::Read,),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_compute_node_with_field(scheme, cells, SLOT_B, GoldyNodeAccess::Overwrite,),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_compute_node_dispatch(scheme, GRID_WIDTH.div_ceil(8), GRID_HEIGHT.div_ceil(8), 1),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let rt = goldy_scheme_lease_render_target(
            scheme,
            GRID_WIDTH,
            GRID_HEIGHT,
            GoldyTextureFormat::Rgba8Unorm,
            false,
            GoldyDepthFormat::Depth24Plus,
        );
        assert!(!rt.is_null(), "{}", last_ffi_message());

        let render_label = CString::new("game_of_life_render").unwrap();
        assert_eq!(
            goldy_scheme_render_pass_begin(
                scheme,
                render_label.as_ptr(),
                rt,
                GoldyTargetLoad::Discard,
                GoldyColor {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
            ),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_render_pass_with_field(scheme, cells, SLOT_B, GoldyNodeAccess::Read),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_render_pass_set_pipeline(scheme, render_pipeline),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_render_pass_draw_fullscreen(scheme),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert_eq!(
            goldy_scheme_render_pass_finish(scheme),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        assert_eq!(
            goldy_scheme_copy_to_texture(scheme, rt, readback),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let grant = goldy_scheme_grant_read_texture(scheme, readback);
        assert!(!grant.is_null(), "{}", last_ffi_message());

        let mut submission = std::ptr::null_mut();
        assert_eq!(
            goldy_scheme_submit(scheme, &mut submission),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        assert!(!submission.is_null());

        let view_size = goldy_buffer_unit_byte_size(cells, SLOT_B) as usize;
        let mut cell_readback = vec![0u8; view_size];
        assert_eq!(
            goldy_buffer_unit_read_to_cpu(cells, SLOT_B, device, cell_readback.as_mut_ptr(), cell_readback.len()),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );
        let cells_out: &[u32] =
            std::slice::from_raw_parts(cell_readback.as_ptr() as *const u32, cell_readback.len() / cell_bytes);
        assert_eq!(
            count_live(cells_out),
            4,
            "still-life block should remain 4 live cells after one step"
        );

        let mut pixels = vec![0u8; (GRID_WIDTH * GRID_HEIGHT * 4) as usize];
        assert_eq!(
            goldy_read_grant_consume(grant, submission, pixels.as_mut_ptr(), pixels.len()),
            GoldyResult::Ok,
            "{}",
            last_ffi_message()
        );

        let cx = (GRID_WIDTH / 2) as usize;
        let cy = (GRID_HEIGHT / 2) as usize;
        let stride = (GRID_WIDTH * 4) as usize;
        let g = pixels[cy * stride + cx * 4 + 1];
        assert!(
            g > 100,
            "center pixel should show alive green channel, got g={g} at ({cx},{cy})"
        );

        goldy_scheme_submission_destroy(submission);
        goldy_read_grant_destroy(grant);
        goldy_scheme_render_target_lease_destroy(rt);
        goldy_scheme_destroy(scheme);
        goldy_render_pipeline_destroy(render_pipeline);
        goldy_compute_pipeline_destroy(compute_pipeline);
        goldy_shader_destroy(render_shader);
        goldy_shader_destroy(compute_shader);
        goldy_texture_destroy(readback);
        goldy_buffer_destroy(cells);
        goldy_retained_pool_destroy(pool);
        goldy_context_destroy(ctx);
        goldy_device_destroy(device);
        goldy_instance_destroy(instance);
    }
}
